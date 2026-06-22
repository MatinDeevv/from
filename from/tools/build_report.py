#!/usr/bin/env python3
# =============================================================================
# build_report.py  —  SUPER-ADVANCED committee report for the XAUUSD model fleet
# -----------------------------------------------------------------------------
# Consumes a walk-forward model registry and emits a SINGLE self-contained
# report.html (no external CSS/JS/fonts, no images on disk: every chart is an
# inline hand-rolled <svg>) plus an executive_summary.md.
#
# It is statistically rigorous and *adversarial*: it deflates every holdout
# statistic against an empirical ZERO-EDGE null distribution, applies a
# Holm-Bonferroni family-wise correction AND a Benjamini-Hochberg FDR control
# across the whole fleet, and is explicit about the optimistic cost model and
# out-of-sample decay so a committee is NOT fooled by noise or overfitting.
#
# PURE PYTHON 3 STDLIB ONLY. numpy/pandas/matplotlib are NEVER imported.
# Robust to a partial / in-progress / sparse / malformed registry and to every
# optional input being absent (degrades to manifest-only).
#
# Inputs (all optional except the registry dir):
#   <registry>/manifest.csv                       (one row per model; canonical)
#   <registry>/<id>/meta.json                     (full nested holdout/oof metrics)
#   <registry>/<id>/report.txt                    (per-fold table + EDGE line)
#   --null PATH    null_results.csv (file,pf,edge,t,neff)  OR a directory of
#                  null *.txt walk-forward reports (per-file EDGE: lines parsed)
#   master_models.csv / factor_effects.csv / interactions.csv / decisions.csv
#                  if present in <registry> (produced by sibling analysis tools);
#                  otherwise the report computes its own factor/interaction views.
#
# Usage:
#   build_report.py <registry_dir> [--null PATH] [--out report.html]
#                   [--top N] [--md executive_summary.md]
# =============================================================================
from __future__ import annotations

import csv
import html
import json
import math
import os
import re
import sys

# -----------------------------------------------------------------------------
# Optional-dependency policy: we DO NOT need numpy/pandas/matplotlib. If present
# they are irrelevant; we never import them. Everything below is stdlib.
# -----------------------------------------------------------------------------

MANIFEST_COLS = [
    "id", "arch", "horizon", "cost_mult", "barrier_k", "conf_gate", "seed",
    "holdout_pf", "holdout_kelly", "holdout_edge", "holdout_t", "holdout_neff",
    "holdout_ci_lo", "oof_pf", "oof_t", "edge_verdict", "dir",
]

# Decision thresholds (committee gate). A model is only interesting if its
# holdout edge survives BOTH the economic gate AND the statistical/null gate.
MIN_NEFF = 30.0          # need >= 30 independent trades to even consider
GO_PF = 1.05             # GO needs a real economic margin after the deflations
WATCH_PF = 1.00          # WATCH is break-even+ but not deflation-significant
GO_KELLY = 0.0
GO_T = 2.0               # crude pre-null t bar


# ----------------------------------------------------------------------------- helpers
def f(v, default=0.0):
    """Best-effort float; tolerant of blanks/junk/None in partial files."""
    try:
        if v is None:
            return default
        s = str(v).strip()
        if s == "" or s.lower() in ("nan", "none", "inf", "-inf"):
            return default
        return float(s)
    except (TypeError, ValueError):
        return default


def i(v, default=0):
    try:
        if v is None or str(v).strip() == "":
            return default
        return int(float(v))
    except (TypeError, ValueError):
        return default


def fmt(v, nd=3):
    try:
        x = float(v)
        if x != x or x in (float("inf"), float("-inf")):
            return "n/a"
        return f"{x:.{nd}f}"
    except (TypeError, ValueError):
        return str(v) if v is not None else "n/a"


def esc(s):
    return html.escape("" if s is None else str(s))


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def stdev(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def percentile(sorted_xs, p):
    """Linear-interpolated percentile, p in [0,1]. sorted_xs must be sorted."""
    if not sorted_xs:
        return float("nan")
    if len(sorted_xs) == 1:
        return sorted_xs[0]
    idx = p * (len(sorted_xs) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_xs[lo]
    frac = idx - lo
    return sorted_xs[lo] * (1 - frac) + sorted_xs[hi] * frac


def normal_cdf(x):
    """Standard normal CDF via erf (stdlib math.erf)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def normal_sf_p(t):
    """One-sided upper-tail p-value P(Z >= t) for a standard normal."""
    return 1.0 - normal_cdf(t)


# ----------------------------------------------------------------------------- loaders
def load_manifest(reg_dir):
    """Load manifest.csv -> list of dict rows (id-keyed, dedup keeping last)."""
    path = os.path.join(reg_dir, "manifest.csv")
    rows = {}
    order = []
    if not os.path.isfile(path):
        return [], path
    try:
        with open(path, "r", newline="", encoding="utf-8", errors="replace") as fh:
            reader = csv.DictReader(fh)
            for raw in reader:
                if raw is None:
                    continue
                rid = (raw.get("id") or "").strip()
                if not rid or rid == "id":
                    continue
                if rid not in rows:
                    order.append(rid)
                rows[rid] = raw  # last wins (latest walk-forward append)
    except OSError:
        return [], path
    return [rows[r] for r in order], path


def load_meta(reg_dir, model_id):
    """Load <id>/meta.json (nested holdout/oof). Returns {} on any failure."""
    p = os.path.join(reg_dir, model_id, "meta.json")
    if not os.path.isfile(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8", errors="replace") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


_FOLD_RE = re.compile(
    r"^\s*fold\s*\d+\b"  # "fold1" or "fold 3" leading token
)
_EDGE_RE = re.compile(
    r"PF=(?P<pf>[-+0-9.eE]+).*?kelly=(?P<kelly>[-+0-9.eE]+).*?"
    r"t=(?P<t>[-+0-9.eE]+).*?N_eff=(?P<neff>[-+0-9.eE]+)"
)
_LABEL_RE = re.compile(r"label dist.*?\[\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*\]")
_NUMTOK = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def parse_report_txt(reg_dir, model_id):
    """Parse <id>/report.txt for per-fold edges, EDGE: line, label dist.

    Per-fold rows look like (whitespace-separated columns):
      fold3  <trades> <N_eff> <winrate> <edge> <PF> <kelly> <maxDD> <se>
             <t_stat> <sharpe> <ci_lo> <ci_hi>
    We robustly pull the numeric columns by position from the first data token
    after the foldN label. Returns dict with fold_edges, fold_t, holdout/oof
    summary, edge line fields, label_dist.
    """
    p = os.path.join(reg_dir, model_id, "report.txt")
    out = {
        "fold_edges": [], "fold_t": [], "fold_pf": [],
        "edge_yes": None, "edge_pf": None, "edge_kelly": None,
        "edge_t": None, "edge_neff": None,
        "label_dist": None, "have": False,
    }
    if not os.path.isfile(p):
        return out
    try:
        with open(p, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except OSError:
        return out
    out["have"] = True
    for ln in lines:
        s = ln.rstrip("\n")
        st = s.strip()
        # per-fold data rows: must start with "foldN" and have many numbers.
        if _FOLD_RE.match(st):
            nums = _NUMTOK.findall(st)
            # The fold index itself is the first integer; skip it. Data columns:
            # trades,N_eff,winrate,edge,PF,kelly,maxDD,se,t_stat,sharpe,ci_lo,ci_hi
            # Drop the leading fold number, then map by position.
            # Some rows are "fold1: steps=.." status lines (no full table) -> skip
            if "steps=" in st or "skipped" in st or "too small" in st:
                continue
            data = nums[1:]  # remove fold index
            if len(data) >= 9:
                edge = f(data[3])
                pf = f(data[4])
                tval = f(data[8])
                out["fold_edges"].append(edge)
                out["fold_pf"].append(pf)
                out["fold_t"].append(tval)
        # EDGE summary line
        if st.startswith("EDGE:"):
            out["edge_yes"] = st.lower().startswith("edge: yes")
            m = _EDGE_RE.search(st)
            if m:
                out["edge_pf"] = f(m.group("pf"))
                out["edge_kelly"] = f(m.group("kelly"))
                out["edge_t"] = f(m.group("t"))
                out["edge_neff"] = f(m.group("neff"))
        # label distribution
        ml = _LABEL_RE.search(st)
        if ml:
            out["label_dist"] = [int(ml.group(1)), int(ml.group(2)), int(ml.group(3))]
    return out


def load_null(null_arg):
    """Return dict with sorted pf/t/neff lists from the null distribution.

    Accepts either a null_results.csv (header file,pf,edge,t,neff) or a directory
    of *.txt walk-forward reports (we parse each EDGE: line). Returns {} if
    nothing usable is found.
    """
    pfs, ts, neffs = [], [], []
    src = None
    if not null_arg:
        return {}
    if os.path.isdir(null_arg):
        src = null_arg
        # prefer a null_results.csv inside the dir if present
        cand = os.path.join(null_arg, "null_results.csv")
        if os.path.isfile(cand):
            return load_null(cand)
        for name in sorted(os.listdir(null_arg)):
            if not name.lower().endswith(".txt"):
                continue
            fp = os.path.join(null_arg, name)
            try:
                with open(fp, "r", encoding="utf-8", errors="replace") as fh:
                    txt = fh.read()
            except OSError:
                continue
            for m in _EDGE_RE.finditer(txt):
                # only count the final holdout EDGE line per file -> last match
                pass
            last = None
            for ln in txt.splitlines():
                if ln.strip().startswith("EDGE:"):
                    last = ln
            if last:
                m = _EDGE_RE.search(last)
                if m:
                    pfs.append(f(m.group("pf")))
                    ts.append(f(m.group("t")))
                    neffs.append(f(m.group("neff")))
    elif os.path.isfile(null_arg):
        src = null_arg
        try:
            with open(null_arg, "r", newline="", encoding="utf-8", errors="replace") as fh:
                reader = csv.DictReader(fh)
                cols = [c.lower() for c in (reader.fieldnames or [])]
                for raw in reader:
                    lc = {(k or "").lower(): v for k, v in raw.items()}
                    if "pf" in lc:
                        pfs.append(f(lc.get("pf")))
                    if "t" in lc:
                        ts.append(f(lc.get("t")))
                    if "neff" in lc:
                        neffs.append(f(lc.get("neff")))
                _ = cols
        except OSError:
            return {}
    else:
        return {}

    pfs = sorted(x for x in pfs if x == x)
    ts = sorted(x for x in ts if x == x)
    neffs = sorted(x for x in neffs if x == x)
    if not pfs and not ts:
        return {}
    return {
        "src": src,
        "pf": pfs, "t": ts, "neff": neffs,
        "pf_p95": percentile(pfs, 0.95) if pfs else float("nan"),
        "pf_p99": percentile(pfs, 0.99) if pfs else float("nan"),
        "pf_mean": mean(pfs) if pfs else float("nan"),
        "t_p95": percentile(ts, 0.95) if ts else float("nan"),
        "t_p99": percentile(ts, 0.99) if ts else float("nan"),
        "t_mean": mean(ts) if ts else float("nan"),
        "n": max(len(pfs), len(ts)),
    }


def load_optional_csv(reg_dir, name):
    """Load an optional sibling CSV into a list of dicts; [] if absent."""
    p = os.path.join(reg_dir, name)
    if not os.path.isfile(p):
        return []
    try:
        with open(p, "r", newline="", encoding="utf-8", errors="replace") as fh:
            return [dict(r) for r in csv.DictReader(fh)]
    except OSError:
        return []


# ----------------------------------------------------------------------------- model assembly
def build_models(reg_dir, manifest_rows):
    """Merge manifest + meta.json + report.txt into one record per model."""
    models = []
    for r in manifest_rows:
        mid = (r.get("id") or "").strip()
        meta = load_meta(reg_dir, mid)
        rep = parse_report_txt(reg_dir, mid)
        mh = (meta.get("metrics", {}) or {}).get("holdout", {}) if meta else {}
        mo = (meta.get("metrics", {}) or {}).get("oof", {}) if meta else {}

        def pick(manifest_key, meta_dict, meta_key):
            mv = meta_dict.get(meta_key) if meta_dict else None
            if mv is not None and str(mv) != "":
                return f(mv)
            return f(r.get(manifest_key))

        rec = {
            "id": mid,
            "arch": (r.get("arch") or meta.get("arch") or "?"),
            "horizon": i(r.get("horizon") or meta.get("horizon")),
            "cost_mult": f(r.get("cost_mult") or meta.get("cost_mult")),
            "barrier_k": f(r.get("barrier_k") or meta.get("barrier_k")),
            "conf_gate": f(r.get("conf_gate") or meta.get("conf_gate")),
            "seed": i(r.get("seed") or meta.get("seed")),
            "h_pf": pick("holdout_pf", mh, "profit_factor"),
            "h_kelly": pick("holdout_kelly", mh, "kelly"),
            "h_edge": pick("holdout_edge", mh, "edge"),
            "h_t": pick("holdout_t", mh, "t_stat"),
            "h_neff": pick("holdout_neff", mh, "n_eff"),
            "h_ci_lo": pick("holdout_ci_lo", mh, "ci_lo"),
            "h_ci_hi": f(mh.get("ci_hi")) if mh else float("nan"),
            "h_winrate": f(mh.get("winrate")) if mh else float("nan"),
            "h_sharpe": f(mh.get("sharpe")) if mh else float("nan"),
            "h_maxdd": f(mh.get("max_drawdown")) if mh else float("nan"),
            "h_trades": i(mh.get("trades")) if mh else 0,
            "oof_pf": pick("oof_pf", mo, "profit_factor"),
            "oof_t": pick("oof_t", mo, "t_stat"),
            "edge_verdict": (r.get("edge_verdict") or meta.get("edge_verdict") or "no").strip().lower(),
            "dir": r.get("dir") or "",
            "fold_edges": rep["fold_edges"],
            "fold_t": rep["fold_t"],
            "fold_pf": rep["fold_pf"],
            "label_dist": rep["label_dist"],
            "have_report": rep["have"],
            "have_meta": bool(meta),
        }
        models.append(rec)
    return models


# ----------------------------------------------------------------------------- statistics
def null_p_value(null, stat_value, kind):
    """Empirical one-sided upper-tail p-value of a model stat vs the null sample.

    p = (1 + #{null >= stat}) / (1 + N)   (add-one / Laplace, never zero).
    Falls back to a normal-tail p on t if no null sample is available.
    """
    arr = null.get(kind) if null else None
    if arr:
        n = len(arr)
        ge = sum(1 for x in arr if x >= stat_value)
        return (1 + ge) / (1 + n)
    if kind == "t":
        return normal_sf_p(stat_value)
    return float("nan")


def holm_bonferroni(pvals):
    """Holm step-down adjusted p-values. Input list of (key, p). Returns dict key->adj_p."""
    items = [(k, p) for k, p in pvals if p == p]
    items.sort(key=lambda kp: kp[1])
    m = len(items)
    adj = {}
    running = 0.0
    for rank, (k, p) in enumerate(items):
        a = min(1.0, (m - rank) * p)
        running = max(running, a)  # enforce monotonicity
        adj[k] = running
    return adj


def benjamini_hochberg(pvals):
    """BH-FDR adjusted p-values (q-values). Input list of (key, p) -> dict key->q."""
    items = [(k, p) for k, p in pvals if p == p]
    items.sort(key=lambda kp: kp[1])
    m = len(items)
    q = {}
    prev = 1.0
    # iterate from largest p to smallest enforcing monotone non-decreasing q
    for rank in range(m - 1, -1, -1):
        k, p = items[rank]
        val = min(prev, p * m / (rank + 1))
        q[k] = val
        prev = val
    return q


def decide(m, null, holm, bh):
    """Committee decision for one model. Returns (decision, reasons:list)."""
    reasons = []
    mid = m["id"]
    pf, kelly, t, neff = m["h_pf"], m["h_kelly"], m["h_t"], m["h_neff"]
    ci_lo = m["h_ci_lo"]
    p_pf = m.get("null_p_pf")
    p_t = m.get("null_p_t")
    holm_p = holm.get(mid, float("nan"))
    q = bh.get(mid, float("nan"))

    # Hard exclusions first.
    if neff < MIN_NEFF:
        return "NO", [f"N_eff={fmt(neff,1)} < {int(MIN_NEFF)} (too few independent trades)"]
    if pf <= 0 or t != t:
        return "NO", ["degenerate / missing holdout stats"]

    # Economic gate.
    econ_go = (pf >= GO_PF and kelly > GO_KELLY and ci_lo > 0)
    econ_watch = (pf >= WATCH_PF and kelly > -0.05)

    # Statistical gate vs null deflation.
    beats_p95 = (null and pf >= null.get("pf_p95", float("inf"))) or \
                (null and t >= null.get("t_p95", float("inf")))
    beats_p99 = (null and pf >= null.get("pf_p99", float("inf"))) or \
                (null and t >= null.get("t_p99", float("inf")))
    null_sig = (p_pf is not None and p_pf == p_pf and p_pf <= 0.05) or \
               (p_t is not None and p_t == p_t and p_t <= 0.05)
    fdr_ok = (q == q and q <= 0.10)
    holm_ok = (holm_p == holm_p and holm_p <= 0.05)

    if econ_go and t >= GO_T and null_sig and (fdr_ok or holm_ok):
        reasons.append(f"PF={fmt(pf,2)}>={GO_PF}, kelly={fmt(kelly,3)}>0, CI_lo={fmt(ci_lo,5)}>0")
        reasons.append(f"t={fmt(t,2)}>={GO_T}, null p_pf={fmt(p_pf,3)} p_t={fmt(p_t,3)}")
        reasons.append(f"survives FDR q={fmt(q,3)} / Holm={fmt(holm_p,3)}")
        if beats_p99:
            reasons.append("exceeds null p99 bar")
        return "GO", reasons

    if econ_watch and (beats_p95 or null_sig or t >= 1.5):
        reasons.append(f"PF={fmt(pf,2)} (break-even+), kelly={fmt(kelly,3)}")
        reasons.append(f"t={fmt(t,2)}, null p_pf={fmt(p_pf,3)}; not deflation-significant")
        if not fdr_ok:
            reasons.append(f"fails FDR (q={fmt(q,3)})")
        return "WATCH", reasons

    # Everything else: NO, with the binding reason.
    if pf < WATCH_PF:
        reasons.append(f"PF={fmt(pf,2)} < {WATCH_PF} (loses money after cost)")
    elif kelly <= 0:
        reasons.append(f"kelly={fmt(kelly,3)} <= 0")
    elif ci_lo <= 0:
        reasons.append(f"edge CI_lo={fmt(ci_lo,5)} <= 0 (not distinguishable from 0)")
    else:
        reasons.append(f"t={fmt(t,2)} fails deflation (null p_pf={fmt(p_pf,3)}, q={fmt(q,3)})")
    return "NO", reasons


# ----------------------------------------------------------------------------- SVG charts
SVG_FONT = "font-family='-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif'"


def _svg_open(w, h, extra=""):
    return (f"<svg viewBox='0 0 {w} {h}' width='100%' "
            f"preserveAspectRatio='xMidYMid meet' "
            f"role='img' style='max-width:{w}px' {extra}>")


def color_for_t(t, tmin, tmax):
    """Diverging blue(neg)-grey(0)-red(pos) for a t value (heatmap cell)."""
    if t != t:
        return "#2a2a33"
    span = max(abs(tmin), abs(tmax), 1e-9)
    x = max(-1.0, min(1.0, t / span))
    if x >= 0:  # positive -> green
        g = int(60 + 150 * x)
        return f"rgb({40},{g},{60})"
    else:       # negative -> red
        rr = int(60 + 150 * (-x))
        return f"rgb({rr},{50},{55})"


def svg_heatmap(models, arch):
    """Heatmap of mean holdout_t over horizon (rows) x cost_mult (cols) for arch."""
    sub = [m for m in models if m["arch"] == arch]
    if not sub:
        return f"<p class='muted'>No <b>{esc(arch)}</b> models yet.</p>"
    horizons = sorted({m["horizon"] for m in sub})
    costs = sorted({round(m["cost_mult"], 3) for m in sub})
    if not horizons or not costs:
        return f"<p class='muted'>No grid data for {esc(arch)}.</p>"
    cell = {}
    for hh in horizons:
        for cc in costs:
            vals = [m["h_t"] for m in sub
                    if m["horizon"] == hh and round(m["cost_mult"], 3) == cc
                    and m["h_t"] == m["h_t"]]
            cell[(hh, cc)] = mean(vals) if vals else float("nan")
    allv = [v for v in cell.values() if v == v]
    tmin = min(allv) if allv else -1.0
    tmax = max(allv) if allv else 1.0

    cw, ch = 78, 30
    left, top = 78, 46
    w = left + cw * len(costs) + 16
    h = top + ch * len(horizons) + 30
    out = [_svg_open(w, h)]
    out.append(f"<text x='8' y='18' {SVG_FONT} font-size='13' font-weight='600' "
               f"fill='#e8e8ef'>arch={esc(arch)}: mean holdout t</text>")
    # column headers (cost_mult)
    for ci, cc in enumerate(costs):
        x = left + ci * cw + cw / 2
        out.append(f"<text x='{x:.0f}' y='{top-6}' {SVG_FONT} font-size='10' "
                   f"text-anchor='middle' fill='#9aa'>cm {fmt(cc,1)}</text>")
    # rows
    for ri, hh in enumerate(horizons):
        y = top + ri * ch
        out.append(f"<text x='{left-6}' y='{y+ch/2+3:.0f}' {SVG_FONT} font-size='10' "
                   f"text-anchor='end' fill='#9aa'>h{hh}</text>")
        for ci, cc in enumerate(costs):
            v = cell[(hh, cc)]
            x = left + ci * cw
            fill = color_for_t(v, tmin, tmax)
            out.append(f"<rect x='{x}' y='{y}' width='{cw-2}' height='{ch-2}' "
                       f"rx='3' fill='{fill}' stroke='#15151c'/>")
            lab = "·" if v != v else fmt(v, 2)
            tc = "#f4f4f8" if (v == v and abs(v) > (max(abs(tmin), abs(tmax)) * 0.4)) else "#cfcfda"
            out.append(f"<text x='{x+cw/2-1:.0f}' y='{y+ch/2+3:.0f}' {SVG_FONT} "
                       f"font-size='10' text-anchor='middle' fill='{tc}'>{lab}</text>")
    out.append("</svg>")
    return "".join(out)


def svg_bar_by_horizon(models):
    """Bar chart of mean holdout_t by horizon (all archs pooled)."""
    horizons = sorted({m["horizon"] for m in models if m["horizon"]})
    if not horizons:
        return "<p class='muted'>No horizon data.</p>"
    data = []
    for hh in horizons:
        vals = [m["h_t"] for m in models if m["horizon"] == hh and m["h_t"] == m["h_t"]]
        data.append((hh, mean(vals) if vals else float("nan"), len(vals)))
    vmax = max([abs(v) for _, v, _ in data if v == v] + [0.5])
    bw = 46
    gap = 14
    left = 44
    plot_h = 150
    w = left + len(data) * (bw + gap) + 16
    h = plot_h + 70
    mid = 30 + plot_h / 2
    out = [_svg_open(w, h)]
    out.append(f"<text x='8' y='18' {SVG_FONT} font-size='13' font-weight='600' "
               f"fill='#e8e8ef'>Mean holdout t by horizon</text>")
    # zero axis
    out.append(f"<line x1='{left}' y1='{mid:.0f}' x2='{w-8}' y2='{mid:.0f}' "
               f"stroke='#44444f' stroke-width='1'/>")
    out.append(f"<text x='{left-6}' y='{mid+3:.0f}' {SVG_FONT} font-size='9' "
               f"text-anchor='end' fill='#888'>0</text>")
    # null t bars reference lines drawn by caller overlay if needed
    for k, (hh, v, n) in enumerate(data):
        x = left + k * (bw + gap)
        if v != v:
            out.append(f"<text x='{x+bw/2:.0f}' y='{mid:.0f}' {SVG_FONT} font-size='9' "
                       f"text-anchor='middle' fill='#666'>n/a</text>")
        else:
            bh_px = (abs(v) / vmax) * (plot_h / 2)
            if v >= 0:
                y = mid - bh_px
                fill = "#3ba776"
            else:
                y = mid
                fill = "#c2564f"
            out.append(f"<rect x='{x}' y='{y:.1f}' width='{bw}' height='{bh_px:.1f}' "
                       f"rx='2' fill='{fill}'/>")
            ty = (y - 4) if v >= 0 else (y + bh_px + 11)
            out.append(f"<text x='{x+bw/2:.0f}' y='{ty:.0f}' {SVG_FONT} font-size='9' "
                       f"text-anchor='middle' fill='#cfcfda'>{fmt(v,2)}</text>")
        out.append(f"<text x='{x+bw/2:.0f}' y='{30+plot_h+16:.0f}' {SVG_FONT} "
                   f"font-size='9' text-anchor='middle' fill='#9aa'>h{hh}</text>")
        out.append(f"<text x='{x+bw/2:.0f}' y='{30+plot_h+28:.0f}' {SVG_FONT} "
                   f"font-size='8' text-anchor='middle' fill='#666'>n={n}</text>")
    out.append("</svg>")
    return "".join(out)


def svg_null_overlay(models, null):
    """Overlay histogram of REAL holdout_pf vs NULL pf, with p95/p99 lines."""
    real = sorted(m["h_pf"] for m in models if m["h_pf"] == m["h_pf"] and m["h_pf"] > 0)
    nul = null.get("pf", []) if null else []
    if not real and not nul:
        return "<p class='muted'>No PF data to plot.</p>"
    lo = 0.0
    hi = max((real[-1] if real else 1.0), (nul[-1] if nul else 1.0), 1.6)
    hi = min(hi, 3.0)  # clip extreme null tails for readability
    nb = 18
    width = (hi - lo) / nb

    def hist(xs):
        b = [0] * nb
        for x in xs:
            if x < lo:
                continue
            k = int((x - lo) / width)
            if k >= nb:
                k = nb - 1
            b[k] += 1
        return b

    rb = hist(real)
    nbk = hist(nul)
    # normalize to densities (fraction) so different N are comparable
    rs = sum(rb) or 1
    ns = sum(nbk) or 1
    rd = [x / rs for x in rb]
    nd = [x / ns for x in nbk]
    dmax = max(rd + nd + [1e-9])

    left, top = 44, 40
    plot_w, plot_h = 460, 150
    w = left + plot_w + 16
    h = top + plot_h + 54
    out = [_svg_open(w, h)]
    out.append(f"<text x='8' y='18' {SVG_FONT} font-size='13' font-weight='600' "
               f"fill='#e8e8ef'>Holdout PF: REAL fleet vs ZERO-EDGE null</text>")
    # axes
    base = top + plot_h
    out.append(f"<line x1='{left}' y1='{base}' x2='{left+plot_w}' y2='{base}' stroke='#44444f'/>")
    bw = plot_w / nb
    for k in range(nb):
        x = left + k * bw
        # null (grey, behind)
        nh = (nd[k] / dmax) * plot_h
        out.append(f"<rect x='{x:.1f}' y='{base-nh:.1f}' width='{bw-1:.1f}' height='{nh:.1f}' "
                   f"fill='#5a5a6e' opacity='0.75'/>")
        # real (teal, front, slightly narrower)
        rh = (rd[k] / dmax) * plot_h
        out.append(f"<rect x='{x+bw*0.22:.1f}' y='{base-rh:.1f}' width='{bw*0.56:.1f}' "
                   f"height='{rh:.1f}' fill='#39a0c9' opacity='0.95'/>")
    # PF=1 reference
    x1 = left + ((1.0 - lo) / (hi - lo)) * plot_w
    out.append(f"<line x1='{x1:.1f}' y1='{top}' x2='{x1:.1f}' y2='{base}' "
               f"stroke='#888' stroke-dasharray='3,3'/>")
    out.append(f"<text x='{x1:.0f}' y='{top-3}' {SVG_FONT} font-size='9' "
               f"text-anchor='middle' fill='#aaa'>PF=1</text>")
    # null p95 / p99
    for q, lab, col in ((null.get("pf_p95"), "p95", "#e0a33e"),
                        (null.get("pf_p99"), "p99", "#e05a5a")):
        if q is not None and q == q and lo <= q <= hi:
            xq = left + ((q - lo) / (hi - lo)) * plot_w
            out.append(f"<line x1='{xq:.1f}' y1='{top}' x2='{xq:.1f}' y2='{base}' "
                       f"stroke='{col}' stroke-width='1.5'/>")
            out.append(f"<text x='{xq:.0f}' y='{base+24}' {SVG_FONT} font-size='9' "
                       f"text-anchor='middle' fill='{col}'>null {lab}={fmt(q,2)}</text>")
    # x ticks
    for tk in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        if tk > hi:
            break
        xt = left + ((tk - lo) / (hi - lo)) * plot_w
        out.append(f"<text x='{xt:.0f}' y='{base+12}' {SVG_FONT} font-size='8' "
                   f"text-anchor='middle' fill='#888'>{fmt(tk,1)}</text>")
    # legend
    ly = top + 4
    out.append(f"<rect x='{left+plot_w-150}' y='{ly}' width='10' height='10' fill='#39a0c9'/>")
    out.append(f"<text x='{left+plot_w-136}' y='{ly+9}' {SVG_FONT} font-size='9' fill='#cfcfda'>real fleet</text>")
    out.append(f"<rect x='{left+plot_w-70}' y='{ly}' width='10' height='10' fill='#5a5a6e'/>")
    out.append(f"<text x='{left+plot_w-56}' y='{ly+9}' {SVG_FONT} font-size='9' fill='#cfcfda'>null</text>")
    out.append("</svg>")
    return "".join(out)


def svg_sparkline(fold_vals, w=130, h=26):
    """Tiny fold-edge sparkline. Green above 0, red below, dot at last."""
    vals = [v for v in fold_vals if v == v]
    if not vals:
        return "<span class='muted'>—</span>"
    vmax = max(abs(v) for v in vals) or 1e-9
    n = len(vals)
    pad = 3
    mid = h / 2
    step = (w - 2 * pad) / max(1, n - 1)
    pts = []
    for k, v in enumerate(vals):
        x = pad + k * step
        y = mid - (v / vmax) * (mid - pad)
        pts.append((x, y))
    out = [_svg_open(w, h, "style='vertical-align:middle'")]
    out.append(f"<line x1='{pad}' y1='{mid}' x2='{w-pad}' y2='{mid}' stroke='#3a3a44'/>")
    # draw segments colored by sign of the midpoint
    for k in range(1, len(pts)):
        x0, y0 = pts[k - 1]
        x1, y1 = pts[k]
        col = "#3ba776" if (vals[k] >= 0) else "#c2564f"
        out.append(f"<line x1='{x0:.1f}' y1='{y0:.1f}' x2='{x1:.1f}' y2='{y1:.1f}' "
                   f"stroke='{col}' stroke-width='1.5'/>")
    lx, lyv = pts[-1]
    lc = "#3ba776" if vals[-1] >= 0 else "#c2564f"
    out.append(f"<circle cx='{lx:.1f}' cy='{lyv:.1f}' r='2.2' fill='{lc}'/>")
    out.append("</svg>")
    return "".join(out)


def render_factor_table(fe_rows):
    """Render precomputed factor_effects.csv rows as an HTML table grouped by knob."""
    if not fe_rows:
        return ""
    cols = [("knob", "knob"), ("value", "value"), ("n", "n"), ("mean_t", "mean t"),
            ("median_t", "median t"), ("mean_pf", "mean PF"), ("pct_pf_gt1", "% PF>1"),
            ("pct_beats_null", "% beat null"), ("mean_kelly", "mean kelly"),
            ("mean_foldpos", "% folds+")]
    out = ["<div class='scroll'><table><thead><tr>"]
    for _, lbl in cols:
        out.append(f"<th>{esc(lbl)}</th>")
    out.append("</tr></thead><tbody>")
    last_knob = None
    for r in fe_rows:
        knob = (r.get("knob") or "").strip()
        sep = " style='border-top:2px solid #2a2a35'" if (knob != last_knob and last_knob is not None) else ""
        last_knob = knob
        mt = f(r.get("mean_t"))
        cls = "pos" if mt > 0 else ("neg" if mt < 0 else "")
        out.append(f"<tr{sep}>")
        out.append(f"<td>{esc(knob)}</td><td>{esc(r.get('value'))}</td>")
        out.append(f"<td>{esc(r.get('n'))}</td>")
        out.append(f"<td class='{cls}'>{fmt(r.get('mean_t'),3)}</td>")
        out.append(f"<td>{fmt(r.get('median_t'),3)}</td>")
        out.append(f"<td>{fmt(r.get('mean_pf'),3)}</td>")
        out.append(f"<td>{fmt(r.get('pct_pf_gt1'),1)}</td>")
        out.append(f"<td>{fmt(r.get('pct_beats_null'),1)}</td>")
        out.append(f"<td>{fmt(r.get('mean_kelly'),3)}</td>")
        fp = r.get("mean_foldpos")
        out.append(f"<td>{fmt(fp,1) if fp not in (None,'') else '—'}</td>")
        out.append("</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def svg_interaction_heatmap(ix_rows, name):
    """Render one precomputed interaction (interactions.csv) as an SVG heatmap."""
    rows = [r for r in ix_rows if (r.get("interaction") or "").strip() == name]
    if not rows:
        return ""
    rvals, cvals = [], []
    cell = {}
    for r in rows:
        rv = (r.get("row_value") or "").strip()
        cv = (r.get("col_value") or "").strip()
        if rv not in rvals:
            rvals.append(rv)
        if cv not in cvals:
            cvals.append(cv)
        cell[(rv, cv)] = f(r.get("mean_t"))

    def numkey(x):
        try:
            return (0, float(x))
        except ValueError:
            return (1, x)
    rvals.sort(key=numkey)
    cvals.sort(key=numkey)
    allv = [v for v in cell.values() if v == v]
    tmin = min(allv) if allv else -1.0
    tmax = max(allv) if allv else 1.0
    cw, ch = 70, 28
    left, top = 92, 44
    w = left + cw * len(cvals) + 16
    h = top + ch * len(rvals) + 22
    out = [_svg_open(w, h)]
    out.append(f"<text x='8' y='18' {SVG_FONT} font-size='13' font-weight='600' "
               f"fill='#e8e8ef'>{esc(name)}: mean holdout t</text>")
    for ci, cv in enumerate(cvals):
        x = left + ci * cw + cw / 2
        out.append(f"<text x='{x:.0f}' y='{top-6}' {SVG_FONT} font-size='10' "
                   f"text-anchor='middle' fill='#9aa'>{esc(cv)}</text>")
    for ri, rv in enumerate(rvals):
        y = top + ri * ch
        out.append(f"<text x='{left-6}' y='{y+ch/2+3:.0f}' {SVG_FONT} font-size='10' "
                   f"text-anchor='end' fill='#9aa'>{esc(rv)}</text>")
        for ci, cv in enumerate(cvals):
            v = cell.get((rv, cv), float("nan"))
            x = left + ci * cw
            out.append(f"<rect x='{x}' y='{y}' width='{cw-2}' height='{ch-2}' rx='3' "
                       f"fill='{color_for_t(v, tmin, tmax)}' stroke='#15151c'/>")
            lab = "·" if v != v else fmt(v, 2)
            out.append(f"<text x='{x+cw/2-1:.0f}' y='{y+ch/2+3:.0f}' {SVG_FONT} font-size='10' "
                       f"text-anchor='middle' fill='#cfcfda'>{lab}</text>")
    out.append("</svg>")
    return "".join(out)


# ----------------------------------------------------------------------------- decision color
DEC_COLOR = {"GO": "#1e7d4f", "WATCH": "#9a7d12", "NO": "#7a2f2f"}
DEC_BG = {"GO": "#16321f", "WATCH": "#322b10", "NO": "#2c1414"}


# ----------------------------------------------------------------------------- exec content
def compute_exec(models, null, decisions, holm, bh):
    n = len(models)
    n_go = sum(1 for m in models if decisions[m["id"]][0] == "GO")
    n_watch = sum(1 for m in models if decisions[m["id"]][0] == "WATCH")
    n_no = sum(1 for m in models if decisions[m["id"]][0] == "NO")

    ranked = sorted(models, key=lambda m: (
        -1 if m["h_neff"] >= MIN_NEFF else 0,  # demote sub-threshold
        m["h_t"] if m["h_t"] == m["h_t"] else -1e9,
        m["h_pf"] if m["h_pf"] == m["h_pf"] else -1e9,
    ), reverse=True)
    best = ranked[0] if ranked else None

    # Is the null sample large enough to *certify* a GO at FDR q<=0.10? The
    # smallest achievable add-one empirical p is 1/(N_null+1); after BH over the
    # eligible family it must still land <= 0.10. If not, GO is unreachable by
    # construction and we must say so (else the committee misreads "0 GO").
    null_n = null.get("n", 0) if null else 0
    n_elig_est = max(1, sum(1 for m in models if m["h_neff"] >= MIN_NEFF))
    min_p = 1.0 / (null_n + 1) if null_n else 1.0
    min_q = min_p * n_elig_est  # BH q for the single best (rank-1) model
    null_underpowered = bool(null) and (min_q > 0.10)

    any_beats = False
    bar_txt = "no null distribution supplied"
    if null:
        bar_txt = (f"null PF p95={fmt(null.get('pf_p95'),2)} / p99={fmt(null.get('pf_p99'),2)}, "
                   f"null t p95={fmt(null.get('t_p95'),2)} / p99={fmt(null.get('t_p99'),2)} "
                   f"(N_null={null.get('n')})")
        for m in models:
            if m["h_neff"] < MIN_NEFF:
                continue
            if (m["h_pf"] >= null.get("pf_p95", float("inf")) or
                    m["h_t"] >= null.get("t_p95", float("inf"))):
                any_beats = True
                break

    # what works / never works (factor sweep on mean holdout_t, eligible models)
    elig = [m for m in models if m["h_neff"] >= MIN_NEFF and m["h_t"] == m["h_t"]]

    def best_level(key, rounder=lambda x: x):
        groups = {}
        for m in elig:
            groups.setdefault(rounder(m[key]), []).append(m["h_t"])
        if not groups:
            return None
        scored = sorted(((k, mean(v)) for k, v in groups.items()),
                        key=lambda kv: kv[1], reverse=True)
        return scored

    works = []
    never = []
    for key, label, rnd in (
        ("horizon", "horizon", lambda x: x),
        ("cost_mult", "cost_mult", lambda x: round(x, 2)),
        ("conf_gate", "conf_gate", lambda x: round(x, 2)),
        ("arch", "arch", lambda x: x),
    ):
        sc = best_level(key, rnd)
        if sc and len(sc) >= 1:
            top_k, top_v = sc[0]
            bot_k, bot_v = sc[-1]
            works.append((label, top_k, top_v))
            if len(sc) >= 2:
                never.append((label, bot_k, bot_v))

    return {
        "n": n, "n_go": n_go, "n_watch": n_watch, "n_no": n_no,
        "best": best, "any_beats": any_beats, "bar_txt": bar_txt,
        "works": works, "never": never, "n_elig": len(elig),
        "null_underpowered": null_underpowered, "null_n": null_n,
        "min_q": min_q, "n_elig_est": n_elig_est,
    }


def plain_english(ex, null):
    L = []
    if ex["n"] == 0:
        return ["No models in the registry yet — training appears to be in progress. "
                "Re-run this report once manifest.csv has rows."]
    if not ex["any_beats"]:
        if null:
            L.append("NOTHING WORKS YET: not a single model's holdout statistic clears the "
                     "deflated (null p95) bar. Every apparent edge is statistically "
                     "indistinguishable from a zero-edge random series after accounting for "
                     "selection over the grid.")
        else:
            L.append("No null distribution was supplied, so significance is judged only by the "
                     "raw economic/statistical gate. Treat all positives as UNCONFIRMED until "
                     "deflated against the empirical null.")
    else:
        L.append(f"{ex['n_go']} model(s) clear the GO bar and {ex['n_watch']} are on WATCH; "
                 "these survive null deflation and multiple-testing control and warrant a "
                 "forward paper-trading confirmation before any capital.")
    if ex.get("null_underpowered") and ex["n_go"] == 0:
        L.append(f"CAVEAT — NULL UNDERPOWERED: the null has only N={ex['null_n']} samples, so the "
                 f"smallest certifiable FDR q for the single best model is ~{fmt(ex['min_q'],3)} "
                 f"(> 0.10) across {ex['n_elig_est']} eligible candidates. GO is therefore "
                 "unreachable purely on resolution grounds — grow the null (the documented study "
                 "uses ~48; aim higher) before concluding a strong model 'fails'. WATCH models "
                 "here may be genuine but cannot yet be certified.")
    if ex["works"]:
        ww = "; ".join(f"{lbl} {k} (mean t={fmt(v,2)})" for lbl, k, v in ex["works"])
        L.append(f"WHAT LOOKS BEST (highest mean holdout t per factor, eligible models only): {ww}.")
    if ex["never"]:
        nn = "; ".join(f"{lbl} {k} (mean t={fmt(v,2)})" for lbl, k, v in ex["never"])
        L.append(f"WHAT NEVER WORKS (worst factor level): {nn}.")
    L.append("All edges are after an OPTIMISTIC cost model; real slippage and OOS decay will "
             "erode them further. A negative profit factor at every horizon is the typical "
             "outcome for raw-window models on XAUUSD ticks — consistent with no exploitable "
             "tick-level edge in this configuration space.")
    return L


# ----------------------------------------------------------------------------- HTML
def render_html(reg_dir, models, null, decisions, holm, bh, ex, top_n,
                optional_present, fe_rows=None, ix_rows=None):
    fe_rows = fe_rows or []
    ix_rows = ix_rows or []
    archs = sorted({m["arch"] for m in models}) if models else []
    parts = []
    A = parts.append

    A("<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>")
    A("<meta name='viewport' content='width=device-width,initial-scale=1'>")
    A("<title>XAUUSD Model-Fleet Committee Report</title>")
    A("<style>")
    A("""
:root{--bg:#0e0e13;--panel:#16161e;--panel2:#1c1c26;--ink:#e8e8ef;--mut:#9a9aae;--line:#2a2a35;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
 font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;line-height:1.5;font-size:14px}
.wrap{max-width:1180px;margin:0 auto;padding:26px 20px 80px}
h1{font-size:24px;margin:0 0 2px}h2{font-size:17px;margin:30px 0 10px;border-bottom:1px solid var(--line);padding-bottom:6px}
h3{font-size:14px;margin:18px 0 8px;color:var(--mut);font-weight:600;text-transform:uppercase;letter-spacing:.04em}
.sub{color:var(--mut);font-size:12px;margin-bottom:18px}
.muted{color:var(--mut)}
.exec{background:linear-gradient(180deg,#191922,#13131a);border:1px solid var(--line);border-radius:12px;padding:20px 22px;margin-top:14px}
.kpis{display:flex;flex-wrap:wrap;gap:12px;margin:6px 0 14px}
.kpi{background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:10px 14px;min-width:96px}
.kpi .v{font-size:22px;font-weight:700}.kpi .l{font-size:11px;color:var(--mut);text-transform:uppercase;letter-spacing:.03em}
.pill{display:inline-block;padding:2px 9px;border-radius:999px;font-size:11px;font-weight:700;letter-spacing:.03em}
.bar{height:10px;border-radius:6px;background:var(--panel2);overflow:hidden;display:flex;margin:4px 0 2px}
.bar i{display:block;height:100%}
.cards{display:flex;flex-wrap:wrap;gap:16px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px;flex:1 1 320px}
.best{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin-top:8px}
ul.pe{margin:8px 0 0;padding-left:18px}ul.pe li{margin:5px 0}
table{border-collapse:collapse;width:100%;font-size:12.5px;margin-top:6px}
th,td{padding:6px 8px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{position:sticky;top:0;background:var(--panel2);cursor:pointer;user-select:none;color:var(--mut);font-weight:600}
th:hover{color:var(--ink)}
tbody tr:hover{background:#1a1a24}
.tag{font-weight:700;padding:1px 7px;border-radius:5px;font-size:11px}
.scroll{overflow:auto;border:1px solid var(--line);border-radius:10px;max-height:680px}
.grid2{display:flex;flex-wrap:wrap;gap:18px}
.chartbox{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px;flex:1 1 340px}
.pos{color:#4cc38a}.neg{color:#e0736b}
.foot{margin-top:34px;font-size:12px;color:var(--mut);border-top:1px solid var(--line);padding-top:14px}
.foot li{margin:5px 0}
code{background:#000;padding:1px 5px;border-radius:4px;font-size:11.5px;color:#c9c9d6}
""")
    A("</style></head><body><div class='wrap'>")

    A("<h1>XAUUSD Walk-Forward Model-Fleet — Committee Report</h1>")
    src_bits = [f"registry <code>{esc(reg_dir)}</code>", f"{ex['n']} models"]
    if null:
        src_bits.append(f"null N={null.get('n')} (<code>{esc(str(null.get('src')))}</code>)")
    else:
        src_bits.append("no null supplied")
    if optional_present:
        src_bits.append("analysis CSVs: " + ", ".join(optional_present))
    A(f"<div class='sub'>{' · '.join(src_bits)}</div>")

    # ---- executive summary ----
    A("<div class='exec'>")
    A("<h3 style='margin-top:0'>Executive summary</h3>")
    A("<div class='kpis'>")
    for v, l in ((ex["n"], "models"), (ex["n_go"], "GO"),
                 (ex["n_watch"], "WATCH"), (ex["n_no"], "NO")):
        A(f"<div class='kpi'><div class='v'>{v}</div><div class='l'>{l}</div></div>")
    A("</div>")
    # decision bar
    tot = max(1, ex["n"])
    A("<div class='bar'>")
    for dec, col in (("GO", DEC_COLOR["GO"]), ("WATCH", DEC_COLOR["WATCH"]), ("NO", DEC_COLOR["NO"])):
        cnt = {"GO": ex["n_go"], "WATCH": ex["n_watch"], "NO": ex["n_no"]}[dec]
        pct = 100.0 * cnt / tot
        if pct > 0:
            A(f"<i style='width:{pct:.2f}%;background:{col}' title='{dec}: {cnt}'></i>")
    A("</div>")

    # best model + deflated bar
    b = ex["best"]
    if b:
        dec, _ = decisions[b["id"]]
        pcol = DEC_COLOR[dec]
        A("<div class='best'>")
        A(f"<b>Best model:</b> <code>{esc(b['id'])}</code> "
          f"<span class='pill' style='background:{DEC_BG[dec]};color:{pcol};border:1px solid {pcol}'>{dec}</span><br>")
        A(f"holdout PF=<b>{fmt(b['h_pf'],2)}</b>, t=<b>{fmt(b['h_t'],2)}</b>, "
          f"kelly=<b>{fmt(b['h_kelly'],3)}</b>, N_eff={fmt(b['h_neff'],1)}")
        if null:
            A(f" · null p(PF)={fmt(b.get('null_p_pf'),3)}, p(t)={fmt(b.get('null_p_t'),3)}, "
              f"FDR q={fmt(bh.get(b['id']),3)}, Holm={fmt(holm.get(b['id']),3)}")
        A("</div>")
    A(f"<p style='margin:10px 0 2px'><b>Deflated bar:</b> {esc(ex['bar_txt'])}. "
      f"<b>Any model beats it?</b> "
      f"<span class='{'pos' if ex['any_beats'] else 'neg'}'>"
      f"{'YES' if ex['any_beats'] else 'NO'}</span></p>")
    # plain english
    A("<h3>What works / what never works</h3><ul class='pe'>")
    for line in plain_english(ex, null):
        A(f"<li>{esc(line)}</li>")
    A("</ul></div>")

    # ---- charts ----
    A("<h2>Diagnostics</h2>")
    A("<div class='grid2'>")
    A("<div class='chartbox'>" + svg_null_overlay(models, null or {}) + "</div>")
    A("<div class='chartbox'>" + svg_bar_by_horizon(models) + "</div>")
    A("</div>")
    A("<h3>Mean holdout t — horizon × cost_mult heatmap, per arch</h3>")
    A("<div class='grid2'>")
    for a in archs:
        A("<div class='chartbox'>" + svg_heatmap(models, a) + "</div>")
    if not archs:
        A("<p class='muted'>No models to chart yet.</p>")
    A("</div>")

    # ---- precomputed factor effects / interactions (sibling CSVs) ----
    if fe_rows or ix_rows:
        A("<h2>Factor effects <span class='muted' style='font-size:12px'>"
          "(precomputed by factor_effects.py)</span></h2>")
        if ix_rows:
            names = []
            for r in ix_rows:
                nm = (r.get("interaction") or "").strip()
                if nm and nm not in names:
                    names.append(nm)
            A("<div class='grid2'>")
            for nm in names:
                svg = svg_interaction_heatmap(ix_rows, nm)
                if svg:
                    A("<div class='chartbox'>" + svg + "</div>")
            A("</div>")
        if fe_rows:
            A("<h3>Marginal effect of each knob (collapsing all other knobs)</h3>")
            A(render_factor_table(fe_rows))

    # ---- top table ----
    ranked = sorted(models, key=lambda m: (
        0 if m["h_neff"] >= MIN_NEFF else 1,
        -(m["h_t"] if m["h_t"] == m["h_t"] else -1e9),
        -(m["h_pf"] if m["h_pf"] == m["h_pf"] else -1e9),
    ))
    show = ranked[:top_n]
    A(f"<h2>Top {len(show)} models <span class='muted' style='font-size:12px'>"
      f"(click a header to sort)</span></h2>")
    cols = [
        ("id", "id", "s"), ("dec", "decision", "s"), ("arch", "arch", "s"),
        ("h", "horizon", "n"), ("cm", "cost", "n"), ("cg", "gate", "n"),
        ("seed", "seed", "n"), ("pf", "PF", "n"), ("t", "t", "n"),
        ("kelly", "kelly", "n"), ("edge", "edge", "n"), ("neff", "N_eff", "n"),
        ("cilo", "CI_lo", "n"), ("ppf", "null p(PF)", "n"), ("pt", "null p(t)", "n"),
        ("q", "FDR q", "n"), ("holm", "Holm", "n"), ("oft", "oof_t", "n"),
        ("fold", "fold edges", "s"),
    ]
    A("<div class='scroll'><table id='tbl'><thead><tr>")
    for k, lbl, typ in cols:
        A(f"<th data-t='{typ}'>{esc(lbl)}</th>")
    A("</tr></thead><tbody>")
    for m in show:
        dec, reasons = decisions[m["id"]]
        pcol = DEC_COLOR[dec]
        title = esc("; ".join(reasons))
        pf_cls = "pos" if m["h_pf"] >= 1.0 else "neg"
        t_cls = "pos" if m["h_t"] >= 2.0 else ("neg" if m["h_t"] < 0 else "")
        spark = svg_sparkline(m["fold_edges"]) if m["fold_edges"] else "<span class='muted'>—</span>"
        A("<tr>")
        A(f"<td>{esc(m['id'])}</td>")
        A(f"<td data-v='{dec}'><span class='tag' title=\"{title}\" "
          f"style='background:{DEC_BG[dec]};color:{pcol};border:1px solid {pcol}'>{dec}</span></td>")
        A(f"<td>{esc(m['arch'])}</td>")
        A(f"<td data-v='{m['horizon']}'>{m['horizon']}</td>")
        A(f"<td data-v='{m['cost_mult']}'>{fmt(m['cost_mult'],1)}</td>")
        A(f"<td data-v='{m['conf_gate']}'>{fmt(m['conf_gate'],2)}</td>")
        A(f"<td data-v='{m['seed']}'>{m['seed']}</td>")
        A(f"<td data-v='{m['h_pf']}' class='{pf_cls}'>{fmt(m['h_pf'],2)}</td>")
        A(f"<td data-v='{m['h_t']}' class='{t_cls}'>{fmt(m['h_t'],2)}</td>")
        A(f"<td data-v='{m['h_kelly']}'>{fmt(m['h_kelly'],3)}</td>")
        A(f"<td data-v='{m['h_edge']}'>{fmt(m['h_edge'],5)}</td>")
        A(f"<td data-v='{m['h_neff']}'>{fmt(m['h_neff'],1)}</td>")
        A(f"<td data-v='{m['h_ci_lo']}'>{fmt(m['h_ci_lo'],5)}</td>")
        A(f"<td data-v='{m.get('null_p_pf','')}'>{fmt(m.get('null_p_pf'),3)}</td>")
        A(f"<td data-v='{m.get('null_p_t','')}'>{fmt(m.get('null_p_t'),3)}</td>")
        A(f"<td data-v='{bh.get(m['id'],'')}'>{fmt(bh.get(m['id']),3)}</td>")
        A(f"<td data-v='{holm.get(m['id'],'')}'>{fmt(holm.get(m['id']),3)}</td>")
        A(f"<td data-v='{m['oof_t']}'>{fmt(m['oof_t'],2)}</td>")
        A(f"<td data-v='0'>{spark}</td>")
        A("</tr>")
    if not show:
        A(f"<tr><td colspan='{len(cols)}' class='muted' style='text-align:center'>"
          "No models in the registry yet (training in progress).</td></tr>")
    A("</tbody></table></div>")

    # ---- GO/WATCH fold sparklines (larger) ----
    gw = [m for m in ranked if decisions[m["id"]][0] in ("GO", "WATCH") and m["fold_edges"]]
    if gw:
        A("<h2>GO / WATCH per-fold edge sparklines</h2><div class='cards'>")
        for m in gw[:12]:
            dec = decisions[m["id"]][0]
            pcol = DEC_COLOR[dec]
            A("<div class='card'>")
            A(f"<div><code>{esc(m['id'])}</code> "
              f"<span class='pill' style='background:{DEC_BG[dec]};color:{pcol};border:1px solid {pcol}'>{dec}</span></div>")
            A("<div style='margin:6px 0'>" + svg_sparkline(m["fold_edges"], w=240, h=44) + "</div>")
            pos = sum(1 for e in m["fold_edges"] if e > 0)
            A(f"<div class='muted' style='font-size:11px'>{pos}/{len(m['fold_edges'])} folds "
              f"edge&gt;0 · holdout PF={fmt(m['h_pf'],2)} t={fmt(m['h_t'],2)}</div>")
            A("</div>")
        A("</div>")

    # ---- methodology / caveats footer ----
    A("<div class='foot'><h2 style='color:var(--ink)'>Methodology &amp; caveats</h2><ul>")
    A("<li><b>Null deflation.</b> Every holdout PF/t is compared to an empirical ZERO-EDGE "
      "distribution generated by running the SAME walk-forward on synthetic series with no "
      "predictable structure (GBM + block-bootstrap of real returns). A model is only "
      "&lsquo;significant&rsquo; if its statistic exceeds the null p95/p99 and its add-one "
      "empirical p-value (p = (1+#{null&ge;stat})/(1+N)) is &le; 0.05.</li>")
    A("<li><b>Multiple testing.</b> With ~324 grid configs, some will look good by chance. We "
      "report BOTH Holm-Bonferroni family-wise adjusted p-values and Benjamini-Hochberg FDR "
      "q-values across the whole fleet; GO requires FDR q&le;0.10 or Holm&le;0.05.</li>")
    A("<li><b>Optimistic cost model.</b> Edges are after a modeled cost of "
      "<code>tb_cost·(1+slippage_mult)+commission</code>. Real fills, partial liquidity, and "
      "adverse selection are worse; treat reported PF as an UPPER bound.</li>")
    A("<li><b>Out-of-sample decay.</b> Walk-forward OOF and a single untouched holdout are the "
      "least-biased estimates available, but live edge typically decays further. GO models must "
      "still pass forward paper-trading before capital.</li>")
    A("<li><b>N_eff, not raw trades.</b> Overlapping windows inflate trade counts; significance "
      f"uses N_eff = trades / overlap-block. Models with N_eff &lt; {int(MIN_NEFF)} are excluded "
      "regardless of PF.</li>")
    A("<li><b>Robustness.</b> This report is pure Python stdlib and degrades gracefully: missing "
      "meta.json / report.txt / null / analysis CSVs are tolerated; manifest.csv alone suffices.</li>")
    A("</ul></div>")

    # ---- sort script (vanilla JS, inline, dependency-free) ----
    A("""
<script>
(function(){
 var tbl=document.getElementById('tbl'); if(!tbl) return;
 var ths=tbl.tHead.rows[0].cells;
 for(var c=0;c<ths.length;c++){(function(idx){
  ths[idx].addEventListener('click',function(){
   var typ=ths[idx].getAttribute('data-t');
   var dir=ths[idx]._d=ths[idx]._d===1?-1:1;
   var rows=Array.prototype.slice.call(tbl.tBodies[0].rows);
   rows.sort(function(a,b){
    var x=a.cells[idx],y=b.cells[idx];
    var xv=x.getAttribute('data-v'), yv=y.getAttribute('data-v');
    if(xv===null) xv=x.textContent; if(yv===null) yv=y.textContent;
    if(typ==='n'){var nx=parseFloat(xv),ny=parseFloat(yv);
      if(isNaN(nx))nx=-Infinity; if(isNaN(ny))ny=-Infinity; return (nx-ny)*dir;}
    return xv.localeCompare(yv)*dir;
   });
   for(var i=0;i<rows.length;i++) tbl.tBodies[0].appendChild(rows[i]);
  });
 })(c);}
})();
</script>
""")
    A("</div></body></html>")
    return "".join(parts)


# ----------------------------------------------------------------------------- markdown
def render_md(reg_dir, models, null, decisions, holm, bh, ex):
    L = []
    L.append("# XAUUSD Model-Fleet — Executive Summary")
    L.append("")
    L.append(f"- **Registry:** `{reg_dir}`")
    L.append(f"- **Total models:** {ex['n']}")
    L.append(f"- **Decisions:** {ex['n_go']} GO · {ex['n_watch']} WATCH · {ex['n_no']} NO")
    if null:
        L.append(f"- **Null distribution:** N={null.get('n')} (`{null.get('src')}`)")
    else:
        L.append("- **Null distribution:** not supplied")
    L.append("")
    b = ex["best"]
    if b:
        dec = decisions[b["id"]][0]
        L.append("## Best model")
        L.append(f"- `{b['id']}` — **{dec}**")
        L.append(f"- holdout PF={fmt(b['h_pf'],2)}, t={fmt(b['h_t'],2)}, "
                 f"kelly={fmt(b['h_kelly'],3)}, N_eff={fmt(b['h_neff'],1)}")
        if null:
            L.append(f"- null p(PF)={fmt(b.get('null_p_pf'),3)}, p(t)={fmt(b.get('null_p_t'),3)}, "
                     f"FDR q={fmt(bh.get(b['id']),3)}, Holm={fmt(holm.get(b['id']),3)}")
        L.append("")
    L.append("## Deflated bar")
    L.append(f"- {ex['bar_txt']}")
    L.append(f"- **Any model beats the deflated bar?** {'YES' if ex['any_beats'] else 'NO'}")
    L.append("")
    L.append("## What works / what never works")
    for line in plain_english(ex, null):
        L.append(f"- {line}")
    L.append("")
    L.append("## Methodology & caveats")
    L.append("- Null deflation vs empirical zero-edge distribution (p95/p99 + add-one empirical p).")
    L.append("- Multiple-testing control: Holm-Bonferroni FWER and Benjamini-Hochberg FDR across the fleet.")
    L.append("- Optimistic cost model — reported PF is an upper bound; real slippage/decay are worse.")
    L.append("- Significance uses N_eff (overlap-adjusted), not raw trade counts; N_eff<30 excluded.")
    L.append("- GO models still require forward paper-trading confirmation before capital.")
    L.append("")
    return "\n".join(L)


# ----------------------------------------------------------------------------- main
def main(argv):
    reg_dir = None
    null_arg = None
    out_html = "report.html"
    out_md = "executive_summary.md"
    top_n = 50
    a = argv[1:]
    k = 0
    while k < len(a):
        tok = a[k]
        if tok == "--null":
            k += 1
            null_arg = a[k] if k < len(a) else None
        elif tok.startswith("--null="):
            null_arg = tok.split("=", 1)[1]
        elif tok == "--out":
            k += 1
            out_html = a[k] if k < len(a) else out_html
        elif tok.startswith("--out="):
            out_html = tok.split("=", 1)[1]
        elif tok == "--md":
            k += 1
            out_md = a[k] if k < len(a) else out_md
        elif tok.startswith("--md="):
            out_md = tok.split("=", 1)[1]
        elif tok == "--top":
            k += 1
            top_n = i(a[k], 50) if k < len(a) else 50
        elif tok.startswith("--top="):
            top_n = i(tok.split("=", 1)[1], 50)
        elif tok in ("-h", "--help"):
            print(__doc__ or "build_report.py <registry_dir> [--null PATH] [--out report.html]")
            return 0
        elif not tok.startswith("-") and reg_dir is None:
            reg_dir = tok
        else:
            sys.stderr.write(f"[build_report] ignoring unknown arg: {tok}\n")
        k += 1

    if reg_dir is None:
        sys.stderr.write("usage: build_report.py <registry_dir> [--null PATH] [--out report.html] "
                         "[--md executive_summary.md] [--top N]\n")
        return 2
    if not os.path.isdir(reg_dir):
        sys.stderr.write(f"[build_report] registry dir not found: {reg_dir}\n")
        return 2

    # default null path: try common locations if not given
    if not null_arg:
        for cand in (os.path.join(reg_dir, "null_results.csv"),
                     os.path.expanduser("~/runs/null/null_results.csv"),
                     os.path.join(os.path.dirname(os.path.abspath(reg_dir)), "runs", "null")):
            if os.path.exists(cand):
                null_arg = cand
                break

    manifest_rows, mpath = load_manifest(reg_dir)
    models = build_models(reg_dir, manifest_rows)
    null = load_null(null_arg) if null_arg else {}

    # null p-values per model
    for m in models:
        m["null_p_pf"] = null_p_value(null, m["h_pf"], "pf") if null else float("nan")
        m["null_p_t"] = null_p_value(null, m["h_t"], "t")  # falls back to normal tail

    # multiple-testing across the fleet on the holdout t null-p (most defensible).
    base_p = []
    for m in models:
        # prefer the null-PF empirical p when a null exists, else t-based p
        p = m["null_p_pf"] if (null and m["null_p_pf"] == m["null_p_pf"]) else m["null_p_t"]
        if m["h_neff"] >= MIN_NEFF and p == p:
            base_p.append((m["id"], p))
    holm = holm_bonferroni(base_p)
    bh = benjamini_hochberg(base_p)

    decisions = {m["id"]: decide(m, null, holm, bh) for m in models}

    # optional sibling CSVs (consumed if present; report notes their presence)
    optional_present = []
    for nm in ("master_models.csv", "factor_effects.csv", "interactions.csv", "decisions.csv"):
        if os.path.isfile(os.path.join(reg_dir, nm)):
            optional_present.append(nm)
    fe_rows = load_optional_csv(reg_dir, "factor_effects.csv")
    ix_rows = load_optional_csv(reg_dir, "interactions.csv")

    ex = compute_exec(models, null, decisions, holm, bh)

    html_doc = render_html(reg_dir, models, null, decisions, holm, bh, ex, top_n,
                           optional_present, fe_rows, ix_rows)
    md_doc = render_md(reg_dir, models, null, decisions, holm, bh, ex)

    try:
        with open(out_html, "w", encoding="utf-8") as fh:
            fh.write(html_doc)
        print(f"[build_report] wrote {out_html}  ({len(html_doc):,} bytes, {ex['n']} models)")
    except OSError as e:
        sys.stderr.write(f"[build_report] could not write {out_html}: {e}\n")
        return 1
    try:
        md_path = out_md
        if os.path.dirname(out_html) and not os.path.dirname(out_md):
            md_path = os.path.join(os.path.dirname(out_html), out_md)
        with open(md_path, "w", encoding="utf-8") as fh:
            fh.write(md_doc)
        print(f"[build_report] wrote {md_path}")
    except OSError as e:
        sys.stderr.write(f"[build_report] could not write {out_md}: {e}\n")

    # console one-liner for orchestration logs
    print(f"[build_report] models={ex['n']} GO={ex['n_go']} WATCH={ex['n_watch']} "
          f"NO={ex['n_no']} any_beats_null={ex['any_beats']} null_N={null.get('n', 0) if null else 0}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
