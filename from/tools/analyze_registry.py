#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""analyze_registry.py — RIGOROUS master results builder for the XAUUSD model fleet.

Pure Python 3 STANDARD LIBRARY only. numpy / pandas / matplotlib are NOT required;
if present they are not used. Runs on CPU. Safe against a still-growing registry
(skips rows / dirs / files it cannot parse) and against NaN / empty / malformed data.

WHAT IT DOES
============
For every model listed in <registry>/manifest.csv (cross-checked with <id>/meta.json and
<id>/report.txt) it computes a statistically deflated, multiple-testing-aware verdict so a
committee can separate real edge from noise and overfitting:

 1. PARSE   holdout + pooled-oof + every per-fold row from report.txt (robust whitespace
            parse, tolerant of "skipped" folds and missing files), backstopped by meta.json
            and the manifest row.
 2. NULL    DEFLATION against the empirical ZERO-EDGE distribution (null_results.csv):
            empirical p-value = fraction of null holdout PF >= model PF (and same for t).
            A model "beats_null" iff  holdout_t > null_p95(t)  AND  holdout_pf > null_p95(pf)
            AND holdout_ci_lo > 0.
 3. MULTIPLE TESTING across all N parsed models: Benjamini-Hochberg FDR q-values and
            Holm-Bonferroni adjusted p-values from each model's null p-value (combined
            PF & t p via the larger / more conservative tail). Counts survivors at FDR<0.1
            and Holm<0.05.
 4. FOLD    STABILITY from the per-fold edges: n_folds, frac_folds_positive, mean/std fold
            edge, worst_fold_edge, and a cross-fold t-stat (mean/se of fold edges).
 5. DECISION per model:
            GO    = beats_null AND frac_folds_positive>=0.6 AND holdout_kelly>0
                    AND survives FDR<0.1
            WATCH = beats_null on the point estimate but fails multiple-testing or fold
                    stability (a promising-but-unconfirmed model)
            NO    = everything else

OUTPUTS  (to --out-dir, default = <registry_dir>)
  master_models.csv   one row per model, all parsed + derived fields, sorted by holdout_t desc
  decisions.csv       id,arch,horizon,cost_mult,conf_gate,seed,holdout_pf,holdout_t,null_p,
                      fdr_q,frac_folds_pos,decision
  + a concise stdout summary (counts by decision, best GO models, null p95/p99 used).

CLI
  analyze_registry.py <registry_dir> [--null PATH] [--out-dir DIR]
  default --null = ~/runs/null/null_results.csv  (degrades gracefully if absent)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys

# ----------------------------------------------------------------------------
# Small numeric helpers (NaN / empty safe). We represent "missing" as None.
# ----------------------------------------------------------------------------

def to_float(x):
    """Best-effort float; returns None for blank / non-numeric / nan / inf."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        v = float(x)
        return v if math.isfinite(v) else None
    s = str(x).strip()
    if not s:
        return None
    sl = s.lower()
    if sl in ("nan", "+nan", "-nan", "na", "n/a", "none", "null", "inf",
              "+inf", "-inf", "infinity", "-infinity"):
        return None
    try:
        v = float(s)
    except (ValueError, TypeError):
        return None
    return v if math.isfinite(v) else None


def to_int(x):
    f = to_float(x)
    return int(round(f)) if f is not None else None


def fmt(x, nd=6):
    """Format a number for CSV; empty string for None."""
    if x is None:
        return ""
    if isinstance(x, float):
        if not math.isfinite(x):
            return ""
        return f"{x:.{nd}f}"
    return str(x)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def stdev(xs):
    """Sample standard deviation (ddof=1); None if <2 points."""
    xs = [x for x in xs if x is not None]
    n = len(xs)
    if n < 2:
        return None
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return math.sqrt(var) if var >= 0 else 0.0


def percentile(sorted_vals, p):
    """Linear-interpolation percentile of an already-sorted list; p in [0,1]."""
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = p * (len(sorted_vals) - 1)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return sorted_vals[lo]
    frac = k - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


# ----------------------------------------------------------------------------
# report.txt parsing
# ----------------------------------------------------------------------------
# Per-fold / pooled / holdout data rows have 12 numeric columns after the label:
#   trades N_eff winrate edge PF kelly maxDD se t_stat sharpe ci95_lo ci95_hi
# Real fold rows look like:  "fold1  35744  270.8  0.458  -0.000167 ...".
# We also tolerate "fold 3" (with a space) per the spec. Lines like
#   "fold 1/8: val block [..] too small (<4096) — skipped"   (no data) and
#   "  fold1: steps=224 (epochs~=8.0, ...)"                   (no data) are ignored.
_NUM = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
_DATA12 = re.compile(r"^\s*(?P<label>[A-Za-z_][\w/ ]*?)\s+" + r"\s+".join([f"(?P<c{i}>{_NUM})" for i in range(12)]) + r"\s*$")
_FOLD_LABEL = re.compile(r"^fold\s*0*([0-9]+)$", re.IGNORECASE)

_COLS = ["trades", "n_eff", "winrate", "edge", "profit_factor", "kelly",
         "max_drawdown", "se", "t_stat", "sharpe", "ci_lo", "ci_hi"]


def _parse_data_row(line):
    """Return (label_str, dict-of-12-metrics) if `line` is a 12-col data row, else None."""
    m = _DATA12.match(line)
    if not m:
        return None
    label = m.group("label").strip()
    vals = [to_float(m.group(f"c{i}")) for i in range(12)]
    return label, dict(zip(_COLS, vals))


def parse_report(path):
    """Parse a report.txt. Returns dict with keys: folds (list of metric dicts),
    pooled_oof, holdout, edge_line_verdict, label_dist. Missing parts -> None / []."""
    out = {"folds": [], "pooled_oof": None, "holdout": None,
           "edge_verdict_report": None, "label_dist": None}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except (OSError, IOError):
        return out

    for raw in text.splitlines():
        line = raw.rstrip("\n")
        stripped = line.strip()
        if not stripped:
            continue

        # label dist (all): [u,n,d]
        if "label dist" in stripped and "[" in stripped:
            mm = re.search(r"\[([^\]]*)\]", stripped)
            if mm:
                parts = [to_int(p) for p in mm.group(1).split(",")]
                if any(p is not None for p in parts):
                    out["label_dist"] = parts
            continue

        # EDGE: yes/no (...)
        if stripped.startswith("EDGE:"):
            mm = re.match(r"EDGE:\s*(yes|no)\b", stripped, re.IGNORECASE)
            if mm:
                out["edge_verdict_report"] = mm.group(1).lower()
            continue

        # data rows (fold / pooled_oof / holdout). Skip obvious non-data lines fast.
        if "(" in stripped and ":" in stripped and "=" in stripped:
            # e.g. "fold1: steps=224 (...)" or "holdout: steps=..." — not a data row.
            continue
        parsed = _parse_data_row(line)
        if not parsed:
            continue
        label, metrics = parsed
        low = label.lower()
        fold_m = _FOLD_LABEL.match(label)
        if fold_m:
            metrics["_fold_index"] = int(fold_m.group(1))
            out["folds"].append(metrics)
        elif low == "pooled_oof":
            out["pooled_oof"] = metrics
        elif low == "holdout":
            out["holdout"] = metrics
        # any other 12-col labelled row is ignored
    # de-dup folds by index (keep last), then sort
    if out["folds"]:
        by_idx = {}
        for f in out["folds"]:
            by_idx[f.get("_fold_index", len(by_idx))] = f
        out["folds"] = [by_idx[k] for k in sorted(by_idx)]
    return out


# ----------------------------------------------------------------------------
# meta.json parsing
# ----------------------------------------------------------------------------

def parse_meta(path):
    """Return (config_dict, holdout_metrics, oof_metrics, edge_verdict) — any may be {}/None."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            data = json.load(fh)
    except (OSError, IOError, ValueError):
        return {}, None, None, None
    if not isinstance(data, dict):
        return {}, None, None, None
    metrics = data.get("metrics") or {}
    hold = metrics.get("holdout") if isinstance(metrics, dict) else None
    oof = metrics.get("oof") if isinstance(metrics, dict) else None

    def norm(d):
        if not isinstance(d, dict):
            return None
        # map meta.json metric keys onto our canonical names
        return {
            "trades": to_float(d.get("trades")),
            "n_eff": to_float(d.get("n_eff")),
            "winrate": to_float(d.get("winrate")),
            "edge": to_float(d.get("edge")),
            "profit_factor": to_float(d.get("profit_factor")),
            "kelly": to_float(d.get("kelly")),
            "max_drawdown": to_float(d.get("max_drawdown")),
            "se": to_float(d.get("se")),
            "t_stat": to_float(d.get("t_stat")),
            "sharpe": to_float(d.get("sharpe")),
            "ci_lo": to_float(d.get("ci_lo")),
            "ci_hi": to_float(d.get("ci_hi")),
        }

    return data, norm(hold), norm(oof), data.get("edge_verdict")


# ----------------------------------------------------------------------------
# manifest.csv parsing
# ----------------------------------------------------------------------------

MANIFEST_FIELDS = ["id", "arch", "horizon", "cost_mult", "barrier_k", "conf_gate",
                   "seed", "holdout_pf", "holdout_kelly", "holdout_edge", "holdout_t",
                   "holdout_neff", "holdout_ci_lo", "oof_pf", "oof_t",
                   "edge_verdict", "dir"]


def parse_manifest(path):
    """Yield manifest rows as dicts; tolerant of partial/blank lines."""
    rows = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                if not r:
                    continue
                mid = (r.get("id") or "").strip()
                if not mid or mid == "id":
                    continue
                rows.append(r)
    except (OSError, IOError):
        pass
    return rows


# ----------------------------------------------------------------------------
# NULL distribution
# ----------------------------------------------------------------------------

class NullDist:
    """Empirical zero-edge distribution of holdout PF and t."""

    def __init__(self, pfs, ts):
        self.pf = sorted(pfs)
        self.t = sorted(ts)
        self.n = max(len(self.pf), len(self.t))

    @property
    def ok(self):
        return len(self.pf) > 0 and len(self.t) > 0

    def pctl_pf(self, p):
        return percentile(self.pf, p)

    def pctl_t(self, p):
        return percentile(self.t, p)

    def pval_pf(self, x):
        """Right-tail empirical p: fraction of null PF >= x."""
        if x is None or not self.pf:
            return None
        return sum(1 for v in self.pf if v >= x) / len(self.pf)

    def pval_t(self, x):
        if x is None or not self.t:
            return None
        return sum(1 for v in self.t if v >= x) / len(self.t)


def load_null(path):
    """Load null_results.csv. Reads PF & t by header name (robust to the known
    header/value column-label mismatch). Returns NullDist (possibly empty)."""
    pfs, ts = [], []
    if not path:
        return NullDist(pfs, ts)
    p = os.path.expanduser(path)
    if not os.path.isfile(p):
        return NullDist(pfs, ts)
    try:
        with open(p, "r", encoding="utf-8", errors="replace", newline="") as fh:
            reader = csv.DictReader(fh)
            fields = [f.strip().lower() for f in (reader.fieldnames or [])]
            for r in reader:
                # access case-insensitively
                rl = {(k or "").strip().lower(): v for k, v in r.items()}
                pf = to_float(rl.get("pf"))
                t = to_float(rl.get("t"))
                if pf is not None:
                    pfs.append(pf)
                if t is not None:
                    ts.append(t)
    except (OSError, IOError):
        pass
    return NullDist(pfs, ts)


# ----------------------------------------------------------------------------
# Multiple-testing corrections
# ----------------------------------------------------------------------------

def benjamini_hochberg(pvals):
    """BH FDR q-values. Input list may contain None (-> None out, excluded from n).
    Returns list aligned to input."""
    idx = [i for i, p in enumerate(pvals) if p is not None]
    m = len(idx)
    q = [None] * len(pvals)
    if m == 0:
        return q
    order = sorted(idx, key=lambda i: pvals[i])
    prev = 1.0
    # iterate from largest p (rank m) down to smallest (rank 1)
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        raw = pvals[i] * m / rank
        prev = min(prev, raw)
        q[i] = min(1.0, prev)
    return q


def holm_bonferroni(pvals):
    """Holm-Bonferroni adjusted p-values. Aligned to input (None preserved)."""
    idx = [i for i, p in enumerate(pvals) if p is not None]
    m = len(idx)
    adj = [None] * len(pvals)
    if m == 0:
        return adj
    order = sorted(idx, key=lambda i: pvals[i])
    prev = 0.0
    for rank in range(1, m + 1):
        i = order[rank - 1]
        raw = pvals[i] * (m - rank + 1)
        prev = max(prev, raw)
        adj[i] = min(1.0, prev)
    return adj


# ----------------------------------------------------------------------------
# Per-model assembly
# ----------------------------------------------------------------------------

def pick(*vals):
    """First non-None value."""
    for v in vals:
        if v is not None:
            return v
    return None


def build_model(mrow, registry_dir):
    """Combine manifest row + meta.json + report.txt into one derived record."""
    mid = (mrow.get("id") or "").strip()
    rec = {"id": mid}

    # config from manifest (authoritative for grid coords)
    rec["arch"] = (mrow.get("arch") or "").strip()
    rec["horizon"] = to_int(mrow.get("horizon"))
    rec["cost_mult"] = to_float(mrow.get("cost_mult"))
    rec["barrier_k"] = to_float(mrow.get("barrier_k"))
    rec["conf_gate"] = to_float(mrow.get("conf_gate"))
    rec["seed"] = to_int(mrow.get("seed"))

    # locate the model dir under the actual registry (manifest 'dir' uses container paths)
    mdir = os.path.join(registry_dir, mid)
    meta_path = os.path.join(mdir, "meta.json")
    report_path = os.path.join(mdir, "report.txt")

    cfg, meta_hold, meta_oof, meta_verdict = parse_meta(meta_path)
    rep = parse_report(report_path)
    rec["has_meta"] = os.path.isfile(meta_path)
    rec["has_report"] = os.path.isfile(report_path)

    # fill grid coords from meta if manifest was blank
    if rec["arch"] == "" and isinstance(cfg, dict):
        rec["arch"] = str(cfg.get("arch") or "")
    if rec["horizon"] is None:
        rec["horizon"] = to_int(cfg.get("horizon")) if isinstance(cfg, dict) else None
    if rec["cost_mult"] is None:
        rec["cost_mult"] = to_float(cfg.get("cost_mult")) if isinstance(cfg, dict) else None
    if rec["conf_gate"] is None:
        rec["conf_gate"] = to_float(cfg.get("conf_gate")) if isinstance(cfg, dict) else None
    if rec["seed"] is None:
        rec["seed"] = to_int(cfg.get("seed")) if isinstance(cfg, dict) else None
    if rec["barrier_k"] is None:
        rec["barrier_k"] = to_float(cfg.get("barrier_k")) if isinstance(cfg, dict) else None

    rep_hold = rep["holdout"]
    rep_oof = rep["pooled_oof"]

    # ---- HOLDOUT metrics: prefer report row, then meta, then manifest ----
    def hcol(report_key, meta_key, manifest_key):
        return pick(
            (rep_hold or {}).get(report_key) if rep_hold else None,
            (meta_hold or {}).get(meta_key) if meta_hold else None,
            to_float(mrow.get(manifest_key)) if manifest_key else None,
        )

    rec["holdout_trades"] = hcol("trades", "trades", None)
    rec["holdout_neff"] = hcol("n_eff", "n_eff", "holdout_neff")
    rec["holdout_winrate"] = hcol("winrate", "winrate", None)
    rec["holdout_edge"] = hcol("edge", "edge", "holdout_edge")
    rec["holdout_pf"] = hcol("profit_factor", "profit_factor", "holdout_pf")
    rec["holdout_kelly"] = hcol("kelly", "kelly", "holdout_kelly")
    rec["holdout_maxdd"] = hcol("max_drawdown", "max_drawdown", None)
    rec["holdout_se"] = hcol("se", "se", None)
    rec["holdout_t"] = hcol("t_stat", "t_stat", "holdout_t")
    rec["holdout_sharpe"] = hcol("sharpe", "sharpe", None)
    rec["holdout_ci_lo"] = hcol("ci_lo", "ci_lo", "holdout_ci_lo")
    rec["holdout_ci_hi"] = hcol("ci_hi", "ci_hi", None)

    # ---- OOF (pooled) metrics ----
    def ocol(report_key, meta_key, manifest_key):
        return pick(
            (rep_oof or {}).get(report_key) if rep_oof else None,
            (meta_oof or {}).get(meta_key) if meta_oof else None,
            to_float(mrow.get(manifest_key)) if manifest_key else None,
        )

    rec["oof_trades"] = ocol("trades", "trades", None)
    rec["oof_neff"] = ocol("n_eff", "n_eff", None)
    rec["oof_edge"] = ocol("edge", "edge", None)
    rec["oof_pf"] = ocol("profit_factor", "profit_factor", "oof_pf")
    rec["oof_kelly"] = ocol("kelly", "kelly", None)
    rec["oof_t"] = ocol("t_stat", "t_stat", "oof_t")
    rec["oof_sharpe"] = ocol("sharpe", "sharpe", None)

    rec["edge_verdict"] = pick(
        (mrow.get("edge_verdict") or "").strip().lower() or None,
        (str(meta_verdict).lower() if meta_verdict is not None else None),
        rep["edge_verdict_report"],
    )
    rec["label_dist"] = rep["label_dist"]

    # ---- FOLD STABILITY ----
    fold_edges = [f.get("edge") for f in rep["folds"]]
    fold_edges = [e for e in fold_edges if e is not None]
    rec["n_folds"] = len(fold_edges)
    if fold_edges:
        npos = sum(1 for e in fold_edges if e > 0)
        rec["frac_folds_positive"] = npos / len(fold_edges)
        rec["mean_fold_edge"] = mean(fold_edges)
        rec["std_fold_edge"] = stdev(fold_edges)
        rec["worst_fold_edge"] = min(fold_edges)
        rec["best_fold_edge"] = max(fold_edges)
        sd = rec["std_fold_edge"]
        if sd is not None and sd > 0 and len(fold_edges) >= 2:
            se = sd / math.sqrt(len(fold_edges))
            rec["fold_edge_t"] = rec["mean_fold_edge"] / se if se > 0 else None
        else:
            rec["fold_edge_t"] = None
    else:
        rec["frac_folds_positive"] = None
        rec["mean_fold_edge"] = None
        rec["std_fold_edge"] = None
        rec["worst_fold_edge"] = None
        rec["best_fold_edge"] = None
        rec["fold_edge_t"] = None

    return rec


# ----------------------------------------------------------------------------
# Significance + decision (needs the null + full-fleet context)
# ----------------------------------------------------------------------------

def attach_null_pvals(rec, null):
    h_pf = rec.get("holdout_pf")
    h_t = rec.get("holdout_t")
    rec["null_p_pf"] = null.pval_pf(h_pf) if null.ok else None
    rec["null_p_t"] = null.pval_t(h_t) if null.ok else None
    # combined p-value: the MORE conservative (larger) tail across PF & t.
    ps = [p for p in (rec["null_p_pf"], rec["null_p_t"]) if p is not None]
    rec["null_p"] = max(ps) if ps else None

    if null.ok:
        p95_pf = null.pctl_pf(0.95)
        p95_t = null.pctl_t(0.95)
        ci_lo = rec.get("holdout_ci_lo")
        rec["beats_null"] = bool(
            h_t is not None and p95_t is not None and h_t > p95_t and
            h_pf is not None and p95_pf is not None and h_pf > p95_pf and
            ci_lo is not None and ci_lo > 0
        )
    else:
        rec["beats_null"] = False


def decide(rec):
    beats = rec.get("beats_null", False)
    ffp = rec.get("frac_folds_positive")
    kelly = rec.get("holdout_kelly")
    fdr_q = rec.get("fdr_q")

    fold_ok = (ffp is not None and ffp >= 0.6)
    kelly_ok = (kelly is not None and kelly > 0)
    fdr_ok = (fdr_q is not None and fdr_q < 0.1)

    if beats and fold_ok and kelly_ok and fdr_ok:
        return "GO"
    if beats:
        return "WATCH"
    return "NO"


# ----------------------------------------------------------------------------
# CSV writers
# ----------------------------------------------------------------------------

MASTER_COLUMNS = [
    "id", "arch", "horizon", "cost_mult", "barrier_k", "conf_gate", "seed",
    "has_meta", "has_report",
    "holdout_trades", "holdout_neff", "holdout_winrate", "holdout_edge",
    "holdout_pf", "holdout_kelly", "holdout_maxdd", "holdout_se", "holdout_t",
    "holdout_sharpe", "holdout_ci_lo", "holdout_ci_hi",
    "oof_trades", "oof_neff", "oof_edge", "oof_pf", "oof_kelly", "oof_t", "oof_sharpe",
    "n_folds", "frac_folds_positive", "mean_fold_edge", "std_fold_edge",
    "worst_fold_edge", "best_fold_edge", "fold_edge_t",
    "edge_verdict", "label_dist",
    "null_p_pf", "null_p_t", "null_p", "beats_null",
    "fdr_q", "holm_p", "fdr_survive", "holm_survive",
    "decision",
]

# number of decimals per master column (default 6)
_ND = {
    "horizon": 0, "seed": 0, "holdout_trades": 0, "oof_trades": 0, "n_folds": 0,
    "cost_mult": 2, "barrier_k": 2, "conf_gate": 4,
    "holdout_neff": 2, "oof_neff": 2, "holdout_maxdd": 6, "holdout_sharpe": 4,
    "oof_sharpe": 4, "holdout_winrate": 4, "holdout_pf": 4, "oof_pf": 4,
    "holdout_kelly": 4, "oof_kelly": 4, "holdout_t": 4, "oof_t": 4,
    "frac_folds_positive": 4, "fold_edge_t": 4,
    "null_p_pf": 6, "null_p_t": 6, "null_p": 6, "fdr_q": 6, "holm_p": 6,
}


def cell(rec, col):
    v = rec.get(col)
    if col == "label_dist":
        if isinstance(v, (list, tuple)):
            return "[" + ",".join("" if x is None else str(x) for x in v) + "]"
        return ""
    if isinstance(v, bool):
        return "1" if v else "0"
    if v is None:
        return ""
    if isinstance(v, float):
        return fmt(v, _ND.get(col, 6))
    if isinstance(v, int):
        return str(v)
    return str(v)


def write_master(path, records):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(MASTER_COLUMNS)
        for r in records:
            w.writerow([cell(r, c) for c in MASTER_COLUMNS])


DECISION_COLUMNS = ["id", "arch", "horizon", "cost_mult", "conf_gate", "seed",
                    "holdout_pf", "holdout_t", "null_p", "fdr_q",
                    "frac_folds_pos", "decision"]


def write_decisions(path, records):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(DECISION_COLUMNS)
        for r in records:
            w.writerow([
                r.get("id", ""),
                r.get("arch", ""),
                cell(r, "horizon"),
                cell(r, "cost_mult"),
                cell(r, "conf_gate"),
                cell(r, "seed"),
                cell(r, "holdout_pf"),
                cell(r, "holdout_t"),
                cell(r, "null_p"),
                cell(r, "fdr_q"),
                cell(r, "frac_folds_positive"),
                r.get("decision", ""),
            ])


# ----------------------------------------------------------------------------
# Sort key (NaN/None last for descending holdout_t)
# ----------------------------------------------------------------------------

def sort_key_t_desc(rec):
    t = rec.get("holdout_t")
    has = t is not None
    return (0 if has else 1, -(t if has else 0.0))


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Build the rigorous, null-deflated, FDR-corrected master results table "
                    "for the XAUUSD model fleet (pure stdlib).")
    ap.add_argument("registry_dir", help="path to the registry directory (contains manifest.csv)")
    ap.add_argument("--null", default="~/runs/null/null_results.csv",
                    help="empirical zero-edge null_results.csv (default ~/runs/null/null_results.csv)")
    ap.add_argument("--out-dir", default=None,
                    help="output directory for master_models.csv / decisions.csv "
                         "(default = registry_dir)")
    args = ap.parse_args(argv)

    registry_dir = os.path.expanduser(args.registry_dir)
    out_dir = os.path.expanduser(args.out_dir) if args.out_dir else registry_dir
    manifest_path = os.path.join(registry_dir, "manifest.csv")

    if not os.path.isdir(registry_dir):
        sys.stderr.write(f"ERROR: registry dir not found: {registry_dir}\n")
        return 2
    if not os.path.isfile(manifest_path):
        sys.stderr.write(f"ERROR: manifest.csv not found in {registry_dir}\n")
        return 2
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as e:
        sys.stderr.write(f"ERROR: cannot create out-dir {out_dir}: {e}\n")
        return 2

    # 1) load null distribution
    null = load_null(args.null)

    # 2) parse manifest + each model
    manifest_rows = parse_manifest(manifest_path)
    records = []
    seen = set()
    skipped = 0
    for mrow in manifest_rows:
        mid = (mrow.get("id") or "").strip()
        if not mid or mid in seen:
            skipped += 1
            continue
        seen.add(mid)
        try:
            rec = build_model(mrow, registry_dir)
        except Exception:  # never let one bad model kill the run
            skipped += 1
            continue
        records.append(rec)

    if not records:
        sys.stderr.write("ERROR: no parseable models in manifest.\n")
        return 1

    # 3) null p-values + beats_null
    for rec in records:
        attach_null_pvals(rec, null)

    # 4) multiple-testing across the fleet (use combined null_p)
    pvals = [rec.get("null_p") for rec in records]
    qvals = benjamini_hochberg(pvals)
    hvals = holm_bonferroni(pvals)
    for rec, q, h in zip(records, qvals, hvals):
        rec["fdr_q"] = q
        rec["holm_p"] = h
        rec["fdr_survive"] = (q is not None and q < 0.1)
        rec["holm_survive"] = (h is not None and h < 0.05)

    # 5) decision
    for rec in records:
        rec["decision"] = decide(rec)

    # sort by holdout_t desc (None last)
    records.sort(key=sort_key_t_desc)

    # write outputs
    master_path = os.path.join(out_dir, "master_models.csv")
    decisions_path = os.path.join(out_dir, "decisions.csv")
    write_master(master_path, records)
    write_decisions(decisions_path, records)

    # ----- stdout summary -----
    n = len(records)
    by_dec = {"GO": 0, "WATCH": 0, "NO": 0}
    for r in records:
        by_dec[r.get("decision", "NO")] = by_dec.get(r.get("decision", "NO"), 0) + 1
    n_fdr = sum(1 for r in records if r.get("fdr_survive"))
    n_holm = sum(1 for r in records if r.get("holm_survive"))
    n_beats = sum(1 for r in records if r.get("beats_null"))

    print("=" * 74)
    print(" XAUUSD FLEET - RIGOROUS MASTER ANALYSIS")
    print("=" * 74)
    print(f" registry        : {registry_dir}")
    print(f" out-dir         : {out_dir}")
    print(f" models parsed   : {n}   (manifest rows skipped/dupe: {skipped})")
    nmeta = sum(1 for r in records if r.get("has_meta"))
    nrep = sum(1 for r in records if r.get("has_report"))
    nfold = sum(1 for r in records if (r.get("n_folds") or 0) > 0)
    print(f" with meta.json  : {nmeta}    with report.txt : {nrep}    with per-fold rows : {nfold}")

    print("-" * 74)
    if null.ok:
        print(f" NULL distribution (zero-edge) : n_pf={len(null.pf)} n_t={len(null.t)}  "
              f"src={os.path.expanduser(args.null)}")
        print(f"   PF   p50={fmt(null.pctl_pf(0.5),3)}  p90={fmt(null.pctl_pf(0.9),3)}  "
              f"p95={fmt(null.pctl_pf(0.95),3)}  p99={fmt(null.pctl_pf(0.99),3)}  "
              f"max={fmt(null.pf[-1],3)}")
        print(f"   t    p50={fmt(null.pctl_t(0.5),3)}  p90={fmt(null.pctl_t(0.9),3)}  "
              f"p95={fmt(null.pctl_t(0.95),3)}  p99={fmt(null.pctl_t(0.99),3)}  "
              f"max={fmt(null.t[-1],3)}")
        print("   (a model beats_null iff holdout_t>p95(t) AND holdout_pf>p95(pf) AND ci_lo>0)")
    else:
        print(" NULL distribution : NOT AVAILABLE - null deflation skipped, "
              "beats_null=False for all.")
        print(f"   (looked for: {os.path.expanduser(args.null)})")

    print("-" * 74)
    print(" DECISIONS")
    print(f"   GO    : {by_dec.get('GO', 0)}")
    print(f"   WATCH : {by_dec.get('WATCH', 0)}")
    print(f"   NO    : {by_dec.get('NO', 0)}")
    print(f"   beats_null (point) : {n_beats}    survive FDR<0.1 : {n_fdr}    "
          f"survive Holm<0.05 : {n_holm}")

    gos = [r for r in records if r.get("decision") == "GO"]
    watches = [r for r in records if r.get("decision") == "WATCH"]
    print("-" * 74)
    if gos:
        print(f" BEST GO MODELS (top {min(10, len(gos))} by holdout_t):")
        _print_table(gos[:10])
    else:
        print(" BEST GO MODELS : none.")
        if watches:
            print(f" Top WATCH candidates (top {min(10, len(watches))} by holdout_t):")
            _print_table(watches[:10])
        else:
            # nothing flagged; show the highest holdout_t models for context
            shown = [r for r in records if r.get("holdout_t") is not None][:10]
            if shown:
                print(f" Highest holdout_t models (context; none flagged):")
                _print_table(shown)
    print("-" * 74)
    print(f" WROTE: {master_path}")
    print(f" WROTE: {decisions_path}")
    print("=" * 74)
    return 0


def _print_table(recs):
    hdr = f"   {'id':40s} {'h_t':>7s} {'h_pf':>6s} {'kelly':>7s} {'null_p':>7s} {'fdr_q':>7s} {'ffp':>5s}"
    print(hdr)
    for r in recs:
        print("   {:40s} {:>7s} {:>6s} {:>7s} {:>7s} {:>7s} {:>5s}".format(
            (r.get("id") or "")[:40],
            fmt(r.get("holdout_t"), 2),
            fmt(r.get("holdout_pf"), 2),
            fmt(r.get("holdout_kelly"), 3),
            fmt(r.get("null_p"), 3),
            fmt(r.get("fdr_q"), 3),
            fmt(r.get("frac_folds_positive"), 2),
        ))


if __name__ == "__main__":
    sys.exit(main())
