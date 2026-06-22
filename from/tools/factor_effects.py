#!/usr/bin/env python3
# =============================================================================
# factor_effects.py  —  WHAT WORKS / WHAT NEVER WORKS for the XAU model fleet
# -----------------------------------------------------------------------------
# Quantifies the MARGINAL effect of every tuning knob on out-of-sample edge so a
# committee can see, with statistical discipline, which settings produce real
# signal, which are marginal, and which never work — without being fooled by
# noise or overfitting.
#
# The grid (dense): arch{mlp,deep} x horizon{1024,2048,4096,6144,8192,12288}
#                   x cost_mult{1.0,1.5,2.0} x conf_gate{0.50,0.55,0.60}
#                   x seed{42,137,2718}  = 324 configs.
#
# For each knob in {arch, horizon, cost_mult, conf_gate, seed} we group models by
# that knob's value (collapsing across all other knobs — a true marginal / main
# effect) and report:
#     n, mean & median holdout_t, mean & median holdout_pf, mean holdout_kelly,
#     %% with holdout_pf>1, %% that BEAT the empirical NULL (deflated),
#     mean frac_folds_positive (walk-forward fold robustness).
# Each knob gets a one-line VERDICT: is the effect monotonic, which value is best,
# and — critically — is the knob pure NOISE. seed SHOULD be ~noise (it is a
# robustness control); if seed swings results a lot, that is an OVERFITTING flag.
#
# Also: two-way interaction grids (mean holdout_t) for arch x horizon and
# horizon x cost_mult, the single BEST config family, the WORST (never works),
# and a noise audit across all knobs.
#
# Significance via NULL deflation: the empirical zero-edge distribution from
# ~/runs/null/null_results.csv (deep h8192 on synthetic null series). A model
# "beats null" iff its holdout_t exceeds the null t p95 AND its holdout_pf
# exceeds the null pf p95 (both — PF and t must clear the noise floor).
#
# PURE PYTHON 3 STDLIB ONLY. numpy/pandas/matplotlib are NOT required and may be
# absent; nothing here imports them. Everything runs on CPU.
#
# Usage:
#   python3 factor_effects.py <registry_dir> [--out-dir <dir>] [--null <csv>]
#
# Outputs (in --out-dir, default = <registry_dir>):
#   factor_effects.csv   knob,value,n,mean_t,median_t,mean_pf,pct_pf_gt1,
#                        pct_beats_null,mean_kelly,mean_foldpos
#   interactions.csv     interaction,row_value,col_value,n,mean_t
#   factor_effects.md    human-readable: per-knob verdicts, interaction heatmaps
#                        as text tables, and a WHAT WORKS / WHAT NEVER WORKS block
# =============================================================================
from __future__ import annotations

import csv
import glob
import math
import os
import re
import sys

# -----------------------------------------------------------------------------
# Knobs and their canonical (sorted) value order. horizon/seed are numeric;
# cost_mult/conf_gate are numeric but presented at fixed precision; arch is
# categorical. ORDER matters for monotonicity tests.
# -----------------------------------------------------------------------------
KNOBS = ["arch", "horizon", "cost_mult", "conf_gate", "seed"]
NUMERIC_KNOBS = {"horizon", "cost_mult", "conf_gate", "seed"}
# Knobs whose effect is EXPECTED to be ~zero (robustness controls). A large
# effect here is an overfitting warning, not a discovery.
NOISE_EXPECTED = {"seed"}


# -----------------------------------------------------------------------------
# Small numeric helpers (tolerant of blank / junk / partial manifest rows).
# -----------------------------------------------------------------------------
def _f(row, key, default=None):
    try:
        v = row.get(key, "")
        if v is None or str(v).strip() == "":
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _i(row, key, default=None):
    try:
        v = row.get(key, "")
        if v is None or str(v).strip() == "":
            return default
        return int(float(v))
    except (TypeError, ValueError):
        return default


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def median(xs):
    xs = sorted(x for x in xs if x is not None)
    n = len(xs)
    if n == 0:
        return None
    if n % 2:
        return xs[n // 2]
    return 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def pct(flags):
    """Percent True among non-None flags; None if no usable observations."""
    flags = [f for f in flags if f is not None]
    if not flags:
        return None
    return 100.0 * sum(1 for f in flags if f) / len(flags)


def quantile(sorted_vals, p):
    """Nearest-rank quantile of an already-sorted list. p in [0,1]."""
    if not sorted_vals:
        return None
    if p <= 0:
        return sorted_vals[0]
    if p >= 1:
        return sorted_vals[-1]
    idx = int(math.ceil(p * len(sorted_vals))) - 1
    idx = min(max(idx, 0), len(sorted_vals) - 1)
    return sorted_vals[idx]


def fnum(v, nd=3, na="n/a"):
    """Format a possibly-None number at fixed precision."""
    if v is None:
        return na
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


# -----------------------------------------------------------------------------
# Canonical knob value strings. The manifest stores ugly floats
# (cost_mult=1, 1.5, 2; conf_gate=0.5500000119). Normalize so grouping is stable
# and labels are clean.
# -----------------------------------------------------------------------------
def canon_value(knob, raw):
    if raw is None:
        return "?"
    s = str(raw).strip()
    if s == "":
        return "?"
    if knob == "arch":
        return s.lower()
    if knob == "horizon" or knob == "seed":
        try:
            return str(int(round(float(s))))
        except (TypeError, ValueError):
            return s
    if knob == "cost_mult":
        try:
            return f"{float(s):.1f}"
        except (TypeError, ValueError):
            return s
    if knob == "conf_gate":
        try:
            return f"{float(s):.2f}"
        except (TypeError, ValueError):
            return s
    return s


def sort_key(knob, value):
    """Order knob values: numeric ascending, arch fixed (mlp<deep), else string."""
    if knob == "arch":
        order = {"mlp": 0, "deep": 1}
        return (order.get(value, 99), value)
    if knob in NUMERIC_KNOBS:
        try:
            return (0, float(value))
        except (TypeError, ValueError):
            return (1, value)
    return (0, value)


# -----------------------------------------------------------------------------
# NULL distribution loader. Header is "file,pf,edge,t,neff" but note the
# study's collector writes KELLY into the "edge" slot; we only use the robust
# pf / t columns for the noise floor, so that mislabel does not bite us.
#
# Resolution order:
#   1) explicit --null path
#   2) <registry>/null_results.csv  (if someone copied it in)
#   3) ~/runs/null/null_results.csv
#   4) parse ~/runs/null/null_*.txt EDGE lines (rebuild on the fly)
# Returns dict {pf_p95,pf_p99,t_p95,t_p99,n} or None if no null data found.
# -----------------------------------------------------------------------------
def _percentiles_from(pfs, ts):
    pfs = sorted(v for v in pfs if v is not None)
    ts = sorted(v for v in ts if v is not None)
    if not ts and not pfs:
        return None
    return {
        "n": max(len(pfs), len(ts)),
        "pf_p95": quantile(pfs, 0.95), "pf_p99": quantile(pfs, 0.99),
        "t_p95": quantile(ts, 0.95), "t_p99": quantile(ts, 0.99),
        "t_p50": quantile(ts, 0.50),
    }


def load_null(registry_dir, explicit):
    candidates = []
    if explicit:
        candidates.append(explicit)
    candidates.append(os.path.join(registry_dir, "null_results.csv"))
    candidates.append(os.path.expanduser("~/runs/null/null_results.csv"))

    for path in candidates:
        if path and os.path.isfile(path):
            pfs, ts = [], []
            try:
                with open(path, "r", newline="", encoding="utf-8",
                          errors="replace") as fh:
                    for r in csv.DictReader(fh):
                        pf = _f(r, "pf")
                        tt = _f(r, "t")
                        if pf is not None:
                            pfs.append(pf)
                        if tt is not None:
                            ts.append(tt)
            except OSError:
                continue
            res = _percentiles_from(pfs, ts)
            if res:
                res["source"] = path
                return res

    # Fallback: rebuild from raw null_*.txt EDGE lines.
    glob_dir = os.path.expanduser("~/runs/null")
    txts = sorted(glob.glob(os.path.join(glob_dir, "null_*.txt")))
    if txts:
        pfs, ts = [], []
        for f in txts:
            try:
                txt = open(f, encoding="utf-8", errors="replace").read()
            except OSError:
                continue
            edge_lines = [ln for ln in txt.splitlines() if ln.startswith("EDGE:")]
            if not edge_lines:
                continue
            ln = edge_lines[-1]
            mpf = re.search(r"PF=([-0-9.]+)", ln)
            mt = re.search(r"\bt=([-0-9.]+)", ln)
            if mpf:
                try:
                    pfs.append(float(mpf.group(1)))
                except ValueError:
                    pass
            if mt:
                try:
                    ts.append(float(mt.group(1)))
                except ValueError:
                    pass
        res = _percentiles_from(pfs, ts)
        if res:
            res["source"] = glob_dir + "/null_*.txt"
            return res

    return None


# -----------------------------------------------------------------------------
# master_models.csv loader (optional). If present we trust its beats_null /
# decision columns. Maps id -> {beats_null(bool|None), decision(str|None)}.
# -----------------------------------------------------------------------------
def load_master(registry_dir):
    path = os.path.join(registry_dir, "master_models.csv")
    if not os.path.isfile(path):
        return {}
    out = {}
    try:
        with open(path, "r", newline="", encoding="utf-8", errors="replace") as fh:
            for r in csv.DictReader(fh):
                rid = (r.get("id") or "").strip()
                if not rid:
                    continue
                bn = None
                for k in ("beats_null", "beat_null", "significant", "deflated_sig"):
                    if k in r and str(r[k]).strip() != "":
                        bn = str(r[k]).strip().lower() in ("1", "true", "yes", "y", "t")
                        break
                dec = None
                for k in ("decision", "verdict", "class", "tier"):
                    if k in r and str(r[k]).strip() != "":
                        dec = str(r[k]).strip()
                        break
                out[rid] = {"beats_null": bn, "decision": dec}
    except OSError:
        pass
    return out


# -----------------------------------------------------------------------------
# Per-fold robustness: parse <dir>/report.txt for fold rows and compute the
# fraction of usable folds with positive edge. Fold rows look like (the model
# may print "fold1" or "fold 1/8"; we only keep DATA rows, i.e. those whose
# numeric columns are present and not the "skipped"/"too small" notices).
#   fold3   <trades> <N_eff> <winrate> <edge> <PF> <kelly> <maxDD> <se> <t> ...
# Column 4 (0-based index 3) is edge. Returns frac in [0,1] or None if no usable
# folds (e.g. all folds skipped because horizon too large for the sample).
# -----------------------------------------------------------------------------
FOLD_RE = re.compile(r"^\s*fold\s*\d+\b")


def frac_folds_positive(model_dir):
    rep = os.path.join(model_dir, "report.txt")
    if not os.path.isfile(rep):
        return None
    pos = 0
    tot = 0
    try:
        with open(rep, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not FOLD_RE.match(line):
                    continue
                low = line.lower()
                if "skip" in low or "too small" in low or "steps=" in low:
                    # progress / skip notices, not a data row
                    continue
                # Strip the leading "foldN" / "fold N/K" label, parse numbers.
                rest = re.sub(r"^\s*fold\s*\d+(\s*/\s*\d+)?[:\s]*", "", line)
                toks = rest.split()
                nums = []
                for t in toks:
                    try:
                        nums.append(float(t))
                    except ValueError:
                        break
                # Need at least through the edge column: trades,N_eff,winrate,edge
                if len(nums) < 4:
                    continue
                edge = nums[3]
                tot += 1
                if edge > 0:
                    pos += 1
    except OSError:
        return None
    if tot == 0:
        return None
    return pos / tot


# -----------------------------------------------------------------------------
# Load manifest rows, enrich each with canonical knob values, holdout metrics,
# beats_null (master override else null-deflation), and frac_folds_positive.
# -----------------------------------------------------------------------------
def load_models(registry_dir, null, master):
    manifest = os.path.join(registry_dir, "manifest.csv")
    models = []
    if not os.path.isfile(manifest):
        return models, manifest

    with open(manifest, "r", newline="", encoding="utf-8", errors="replace") as fh:
        for raw in csv.DictReader(fh):
            rid = (raw.get("id") or "").strip()
            if not rid or rid == "id":
                continue

            m = {
                "id": rid,
                "holdout_t": _f(raw, "holdout_t"),
                "holdout_pf": _f(raw, "holdout_pf"),
                "holdout_kelly": _f(raw, "holdout_kelly"),
                "edge_verdict": (raw.get("edge_verdict") or "").strip().lower(),
            }
            for knob in KNOBS:
                m[knob] = canon_value(knob, raw.get(knob))

            # pf>1 flag
            m["pf_gt1"] = (m["holdout_pf"] > 1.0) if m["holdout_pf"] is not None else None

            # beats_null: master override > null-deflation > None
            bn = None
            if rid in master and master[rid].get("beats_null") is not None:
                bn = master[rid]["beats_null"]
            elif null is not None:
                t = m["holdout_t"]
                pf = m["holdout_pf"]
                tp = null.get("t_p95")
                pp = null.get("pf_p95")
                if t is not None and pf is not None and tp is not None and pp is not None:
                    bn = (t > tp) and (pf > pp)
            m["beats_null"] = bn

            # fold robustness from report.txt (resolve dir relative to registry)
            mdir = os.path.join(registry_dir, rid)
            if not os.path.isdir(mdir):
                # fall back to the dir column if present and local
                draw = (raw.get("dir") or "").strip()
                if draw:
                    cand = os.path.join(registry_dir, os.path.basename(draw))
                    if os.path.isdir(cand):
                        mdir = cand
            m["foldpos"] = frac_folds_positive(mdir) if os.path.isdir(mdir) else None

            models.append(m)

    return models, manifest


# -----------------------------------------------------------------------------
# Per-knob grouping + summary statistics.
# -----------------------------------------------------------------------------
def summarize_knob(models, knob):
    groups = {}
    for m in models:
        groups.setdefault(m[knob], []).append(m)
    rows = []
    for value in sorted(groups, key=lambda v: sort_key(knob, v)):
        g = groups[value]
        rows.append({
            "knob": knob,
            "value": value,
            "n": len(g),
            "mean_t": mean([x["holdout_t"] for x in g]),
            "median_t": median([x["holdout_t"] for x in g]),
            "mean_pf": mean([x["holdout_pf"] for x in g]),
            "median_pf": median([x["holdout_pf"] for x in g]),
            "pct_pf_gt1": pct([x["pf_gt1"] for x in g]),
            "pct_beats_null": pct([x["beats_null"] for x in g]),
            "mean_kelly": mean([x["holdout_kelly"] for x in g]),
            "mean_foldpos": mean([x["foldpos"] for x in g]),
        })
    return rows


# -----------------------------------------------------------------------------
# Verdict logic per knob.
#   * range = max(mean_t) - min(mean_t) across this knob's values
#   * monotonic = mean_t is (weakly) non-decreasing OR non-increasing along the
#     ordered values (only meaningful for numeric knobs with >=3 levels)
#   * best value = the one with the highest mean_t
#   * noise: range below NOISE_T threshold => effectively no effect
# For seed (NOISE_EXPECTED): a LARGE range is an OVERFITTING flag.
# -----------------------------------------------------------------------------
NOISE_T = 0.25     # |mean_t| spread under this across a knob's values = noise
SEED_WARN_T = 0.40  # seed spread over this = overfitting warning


def is_monotonic(vals):
    """vals: list of (sortable_key, value_or_None). Returns 'up','down','non' or None."""
    seq = [v for _, v in vals if v is not None]
    if len(seq) < 3:
        return None
    up = all(seq[i] <= seq[i + 1] + 1e-12 for i in range(len(seq) - 1))
    down = all(seq[i] >= seq[i + 1] - 1e-12 for i in range(len(seq) - 1))
    if up and not down:
        return "up"
    if down and not up:
        return "down"
    if up and down:
        return "flat"
    return "non"


def knob_verdict(knob, rows):
    usable = [r for r in rows if r["mean_t"] is not None]
    if not usable:
        return "no holdout_t data for this knob"

    mts = [r["mean_t"] for r in usable]
    rng = max(mts) - min(mts)
    best = max(usable, key=lambda r: r["mean_t"])
    worst = min(usable, key=lambda r: r["mean_t"])

    ordered = [(sort_key(knob, r["value"]), r["mean_t"]) for r in usable]
    ordered.sort(key=lambda x: x[0])
    mono = is_monotonic(ordered)

    # noise classification
    if knob in NOISE_EXPECTED:
        if rng <= NOISE_T:
            return (f"NOISE as expected (mean_t spread={rng:.2f} <= {NOISE_T}); "
                    f"robustness check PASSES — results stable across {knob}.")
        if rng >= SEED_WARN_T:
            return (f"OVERFITTING WARNING — {knob} should not matter but mean_t "
                    f"spread={rng:.2f} (best {knob}={best['value']} t={best['mean_t']:.2f} "
                    f"vs worst={worst['value']} t={worst['mean_t']:.2f}); "
                    f"signal may be seed-specific luck, not real edge.")
        return (f"mild {knob} sensitivity (spread={rng:.2f}); borderline robustness — "
                f"watch for overfitting.")

    if rng <= NOISE_T:
        return (f"PURE NOISE — mean_t spread only {rng:.2f} across values; "
                f"{knob} has no material effect on out-of-sample edge.")

    bits = [f"best {knob}={best['value']} (mean_t={best['mean_t']:.2f}, "
            f"%pf>1={fnum(best['pct_pf_gt1'],0)}%)"]
    if mono == "up":
        bits.append("effect is MONOTONIC INCREASING in value")
    elif mono == "down":
        bits.append("effect is MONOTONIC DECREASING in value")
    elif mono == "non":
        bits.append("effect is non-monotonic (interior optimum / mixed)")
    bits.append(f"spread={rng:.2f}, worst={worst['value']} (t={worst['mean_t']:.2f})")
    return "; ".join(bits) + "."


# -----------------------------------------------------------------------------
# Two-way interaction grid: mean holdout_t for each (row_knob, col_knob) cell.
# -----------------------------------------------------------------------------
def interaction(models, row_knob, col_knob):
    rvals = sorted({m[row_knob] for m in models}, key=lambda v: sort_key(row_knob, v))
    cvals = sorted({m[col_knob] for m in models}, key=lambda v: sort_key(col_knob, v))
    cells = {}
    for m in models:
        cells.setdefault((m[row_knob], m[col_knob]), []).append(m["holdout_t"])
    grid = {}
    for rv in rvals:
        for cv in cvals:
            g = cells.get((rv, cv), [])
            grid[(rv, cv)] = (len([x for x in g if x is not None]),
                              mean([x for x in g if x is not None]))
    return rvals, cvals, grid


def render_grid_md(title, row_knob, col_knob, rvals, cvals, grid):
    out = [f"### Interaction: {row_knob} x {col_knob}  (cell = mean holdout_t)", ""]
    header = f"| {row_knob} \\ {col_knob} | " + " | ".join(str(c) for c in cvals) + " |"
    sep = "|" + "---|" * (len(cvals) + 1)
    out.append(header)
    out.append(sep)
    for rv in rvals:
        cells = []
        for cv in cvals:
            n, mt = grid[(rv, cv)]
            cells.append("n/a" if mt is None else f"{mt:+.2f}")
        out.append(f"| {rv} | " + " | ".join(cells) + " |")
    out.append("")
    return "\n".join(out)


# -----------------------------------------------------------------------------
# Config FAMILY = (arch, horizon) — the two structural knobs. Best / worst family
# by mean holdout_t (with a small-n caveat). Used for WHAT WORKS / NEVER WORKS.
# -----------------------------------------------------------------------------
def family_table(models):
    fam = {}
    for m in models:
        key = (m["arch"], m["horizon"])
        fam.setdefault(key, []).append(m)
    rows = []
    for key, g in fam.items():
        rows.append({
            "family": f"{key[0]}_h{key[1]}",
            "arch": key[0], "horizon": key[1],
            "n": len(g),
            "mean_t": mean([x["holdout_t"] for x in g]),
            "median_t": median([x["holdout_t"] for x in g]),
            "mean_pf": mean([x["holdout_pf"] for x in g]),
            "pct_pf_gt1": pct([x["pf_gt1"] for x in g]),
            "pct_beats_null": pct([x["beats_null"] for x in g]),
            "mean_foldpos": mean([x["foldpos"] for x in g]),
        })
    return rows


# -----------------------------------------------------------------------------
# Output writers.
# -----------------------------------------------------------------------------
def write_factor_csv(path, all_knob_rows):
    cols = ["knob", "value", "n", "mean_t", "median_t", "mean_pf", "pct_pf_gt1",
            "pct_beats_null", "mean_kelly", "mean_foldpos"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in all_knob_rows:
            w.writerow([
                r["knob"], r["value"], r["n"],
                fnum(r["mean_t"], 4, ""), fnum(r["median_t"], 4, ""),
                fnum(r["mean_pf"], 4, ""), fnum(r["pct_pf_gt1"], 1, ""),
                fnum(r["pct_beats_null"], 1, ""), fnum(r["mean_kelly"], 4, ""),
                fnum(r["mean_foldpos"], 4, ""),
            ])


def write_interactions_csv(path, interactions):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["interaction", "row_value", "col_value", "n", "mean_t"])
        for name, rvals, cvals, grid in interactions:
            for rv in rvals:
                for cv in cvals:
                    n, mt = grid[(rv, cv)]
                    w.writerow([name, rv, cv, n, fnum(mt, 4, "")])


# -----------------------------------------------------------------------------
# Main.
# -----------------------------------------------------------------------------
def main(argv):
    # Best-effort UTF-8 console so em-dashes etc. don't mojibake on Windows
    # (cp1252). Harmless no-op on POSIX / older Pythons.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass

    registry_dir = None
    out_dir = None
    null_path = None
    i = 1
    while i < len(argv):
        a = argv[i]
        if a == "--out-dir":
            i += 1
            out_dir = argv[i] if i < len(argv) else None
        elif a.startswith("--out-dir="):
            out_dir = a.split("=", 1)[1]
        elif a == "--null":
            i += 1
            null_path = argv[i] if i < len(argv) else None
        elif a.startswith("--null="):
            null_path = a.split("=", 1)[1]
        elif a in ("-h", "--help"):
            print(__doc__ if __doc__ else "factor_effects.py <registry_dir> [--out-dir D] [--null CSV]")
            return 0
        elif registry_dir is None:
            registry_dir = a
        i += 1

    if not registry_dir:
        print("usage: factor_effects.py <registry_dir> [--out-dir <dir>] [--null <csv>]",
              file=sys.stderr)
        return 2
    registry_dir = os.path.abspath(os.path.expanduser(registry_dir))
    if not os.path.isdir(registry_dir):
        print(f"error: registry dir not found: {registry_dir}", file=sys.stderr)
        return 2
    if out_dir is None:
        out_dir = registry_dir
    out_dir = os.path.abspath(os.path.expanduser(out_dir))
    os.makedirs(out_dir, exist_ok=True)

    null = load_null(registry_dir, null_path)
    master = load_master(registry_dir)
    models, manifest = load_models(registry_dir, null, master)

    if not models:
        print(f"error: no usable models in {manifest}", file=sys.stderr)
        return 1

    # --- per-knob summaries -------------------------------------------------
    knob_summaries = {k: summarize_knob(models, k) for k in KNOBS}
    all_knob_rows = []
    for k in KNOBS:
        all_knob_rows.extend(knob_summaries[k])
    verdicts = {k: knob_verdict(k, knob_summaries[k]) for k in KNOBS}

    # --- interactions -------------------------------------------------------
    ax_r, ax_c, ax_grid = interaction(models, "arch", "horizon")
    hc_r, hc_c, hc_grid = interaction(models, "horizon", "cost_mult")
    interactions = [
        ("arch x horizon", ax_r, ax_c, ax_grid),
        ("horizon x cost_mult", hc_r, hc_c, hc_grid),
    ]

    # --- families: best / worst --------------------------------------------
    fams = family_table(models)
    fams_usable = [f for f in fams if f["mean_t"] is not None]
    best_fam = max(fams_usable, key=lambda f: f["mean_t"]) if fams_usable else None
    worst_fam = min(fams_usable, key=lambda f: f["mean_t"]) if fams_usable else None

    # --- noise audit across all knobs --------------------------------------
    noise_knobs = []
    overfit_knobs = []
    for k in KNOBS:
        mts = [r["mean_t"] for r in knob_summaries[k] if r["mean_t"] is not None]
        if len(mts) < 2:
            continue
        rng = max(mts) - min(mts)
        if k in NOISE_EXPECTED:
            if rng >= SEED_WARN_T:
                overfit_knobs.append((k, rng))
        else:
            if rng <= NOISE_T:
                noise_knobs.append((k, rng))

    # --- write CSVs ---------------------------------------------------------
    fe_csv = os.path.join(out_dir, "factor_effects.csv")
    ix_csv = os.path.join(out_dir, "interactions.csv")
    fe_md = os.path.join(out_dir, "factor_effects.md")
    write_factor_csv(fe_csv, all_knob_rows)
    write_interactions_csv(ix_csv, interactions)

    # --- compose markdown ---------------------------------------------------
    md = []
    md.append("# Factor-Effects Report — XAU Model Fleet")
    md.append("")
    md.append(f"- registry: `{registry_dir}`")
    md.append(f"- models analyzed: **{len(models)}**")
    if null is not None:
        md.append(f"- null floor (deflation): n={null.get('n')} "
                  f"t_p95={fnum(null.get('t_p95'),3)} t_p99={fnum(null.get('t_p99'),3)} "
                  f"pf_p95={fnum(null.get('pf_p95'),3)} pf_p99={fnum(null.get('pf_p99'),3)}  "
                  f"(`{null.get('source')}`)")
        md.append("  - a model **beats null** iff holdout_t > t_p95 AND holdout_pf > pf_p95.")
    else:
        md.append("- null floor: **UNAVAILABLE** — %beats_null reported as n/a "
                  "(no ~/runs/null/null_results.csv and no null_*.txt to rebuild from).")
    if master:
        md.append(f"- master_models.csv: present ({len(master)} rows) — "
                  "beats_null taken from it where available.")
    md.append("")

    # per-knob tables + verdicts
    md.append("## Per-Knob Marginal Effects")
    md.append("")
    for k in KNOBS:
        md.append(f"### Knob: `{k}`")
        md.append("")
        md.append("| value | n | mean_t | median_t | mean_pf | %pf>1 | %beats_null | mean_kelly | mean_foldpos |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for r in knob_summaries[k]:
            md.append(
                f"| {r['value']} | {r['n']} | {fnum(r['mean_t'],3)} | "
                f"{fnum(r['median_t'],3)} | {fnum(r['mean_pf'],3)} | "
                f"{fnum(r['pct_pf_gt1'],0)} | {fnum(r['pct_beats_null'],0)} | "
                f"{fnum(r['mean_kelly'],3)} | {fnum(r['mean_foldpos'],2)} |")
        md.append("")
        md.append(f"**VERDICT ({k}):** {verdicts[k]}")
        md.append("")

    # interaction heatmaps
    md.append("## Two-Way Interactions")
    md.append("")
    md.append(render_grid_md("arch x horizon", "arch", "horizon", ax_r, ax_c, ax_grid))
    md.append(render_grid_md("horizon x cost_mult", "horizon", "cost_mult",
                             hc_r, hc_c, hc_grid))

    # WHAT WORKS / WHAT NEVER WORKS
    md.append("## WHAT WORKS / WHAT NEVER WORKS")
    md.append("")
    if best_fam:
        md.append(f"- **BEST config family:** `{best_fam['family']}` — "
                  f"mean_t={fnum(best_fam['mean_t'],3)}, median_t={fnum(best_fam['median_t'],3)}, "
                  f"mean_pf={fnum(best_fam['mean_pf'],3)}, %pf>1={fnum(best_fam['pct_pf_gt1'],0)}%, "
                  f"%beats_null={fnum(best_fam['pct_beats_null'],0)}%, "
                  f"mean_foldpos={fnum(best_fam['mean_foldpos'],2)} (n={best_fam['n']})")
    if worst_fam:
        md.append(f"- **WORST config family (never works):** `{worst_fam['family']}` — "
                  f"mean_t={fnum(worst_fam['mean_t'],3)}, mean_pf={fnum(worst_fam['mean_pf'],3)}, "
                  f"%pf>1={fnum(worst_fam['pct_pf_gt1'],0)}%, "
                  f"%beats_null={fnum(worst_fam['pct_beats_null'],0)}% (n={worst_fam['n']})")
    # any family that beats null at all?
    any_beat = [f for f in fams_usable if (f["pct_beats_null"] or 0) > 0]
    n_beat_models = sum(1 for m in models if m["beats_null"]) if null is not None else None
    if null is not None:
        md.append(f"- models that **beat the null floor**: "
                  f"{n_beat_models}/{len(models)}"
                  + (f"  (families with any survivor: {len(any_beat)})" if any_beat else
                     "  — NONE; no config family clears the noise floor."))
    if noise_knobs:
        md.append("- **PURE-NOISE knobs** (no material effect on edge): "
                  + ", ".join(f"`{k}` (Δmean_t={rng:.2f})" for k, rng in noise_knobs))
    else:
        md.append("- pure-noise knobs: none detected (every knob moves edge somewhat).")
    if overfit_knobs:
        md.append("- **OVERFITTING FLAGS** (knob that should be inert but isn't): "
                  + ", ".join(f"`{k}` (Δmean_t={rng:.2f})" for k, rng in overfit_knobs))
    else:
        md.append("- overfitting flags (seed sensitivity): none — "
                  "seed is inert as it should be (robustness check passes).")
    md.append("")

    # family ranking table (full, for the committee)
    md.append("### Config-family ranking (by mean holdout_t)")
    md.append("")
    md.append("| family | n | mean_t | median_t | mean_pf | %pf>1 | %beats_null | mean_foldpos |")
    md.append("|---|---|---|---|---|---|---|---|")
    for f in sorted(fams, key=lambda x: (x["mean_t"] is None, -(x["mean_t"] or -1e9))):
        md.append(
            f"| {f['family']} | {f['n']} | {fnum(f['mean_t'],3)} | "
            f"{fnum(f['median_t'],3)} | {fnum(f['mean_pf'],3)} | "
            f"{fnum(f['pct_pf_gt1'],0)} | {fnum(f['pct_beats_null'],0)} | "
            f"{fnum(f['mean_foldpos'],2)} |")
    md.append("")

    md_text = "\n".join(md)
    with open(fe_md, "w", encoding="utf-8") as fh:
        fh.write(md_text + "\n")

    # --- console summary ----------------------------------------------------
    print("=" * 74)
    print(f"  FACTOR-EFFECTS  registry={registry_dir}  models={len(models)}")
    print("=" * 74)
    if null is not None:
        print(f"  null floor: t_p95={fnum(null.get('t_p95'),2)} "
              f"pf_p95={fnum(null.get('pf_p95'),2)} (n={null.get('n')}, "
              f"src={os.path.basename(str(null.get('source')))})")
    else:
        print("  null floor: UNAVAILABLE -> %beats_null = n/a")
    print("  " + "-" * 70)
    print("  PER-KNOB VERDICTS:")
    for k in KNOBS:
        print(f"   [{k}] {verdicts[k]}")
    print("  " + "-" * 70)
    if best_fam:
        print(f"  BEST family : {best_fam['family']:<14} "
              f"mean_t={fnum(best_fam['mean_t'],2)} mean_pf={fnum(best_fam['mean_pf'],2)} "
              f"%pf>1={fnum(best_fam['pct_pf_gt1'],0)}% "
              f"%beats_null={fnum(best_fam['pct_beats_null'],0)}% (n={best_fam['n']})")
    if worst_fam:
        print(f"  WORST family: {worst_fam['family']:<14} "
              f"mean_t={fnum(worst_fam['mean_t'],2)} mean_pf={fnum(worst_fam['mean_pf'],2)} "
              f"%pf>1={fnum(worst_fam['pct_pf_gt1'],0)}% (never works)")
    if null is not None:
        print(f"  models beating null floor: {n_beat_models}/{len(models)}")
    if noise_knobs:
        print("  PURE-NOISE knobs: " + ", ".join(k for k, _ in noise_knobs))
    if overfit_knobs:
        print("  OVERFITTING FLAGS: " + ", ".join(f"{k}(d={r:.2f})" for k, r in overfit_knobs))
    else:
        print("  seed robustness: PASS (seed inert)")
    print("  " + "-" * 70)
    print(f"  wrote: {fe_csv}")
    print(f"  wrote: {ix_csv}")
    print(f"  wrote: {fe_md}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
