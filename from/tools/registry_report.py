#!/usr/bin/env python3
# =============================================================================
# registry_report.py  —  rank + summarize the FLEET artifact registry
# -----------------------------------------------------------------------------
# Reads <registry>/manifest.csv (one row per EDGE-positive walk-forward run, as
# appended by include/io/artifact.hpp::append_manifest_row) and prints:
#   * a ranked table of the top models by holdout t_stat among edge_verdict==yes,
#     ties broken by holdout profit-factor (holdout_pf),
#   * a summary: configs run, edge_verdict==yes count, best holdout PF/kelly/t,
#     and counts broken down by arch and by horizon.
# Also writes <registry>/leaderboard.txt next to the manifest.
#
# Pure stdlib (csv, sys, os). Robust to partial / in-progress / malformed rows.
#
# Usage:
#   python3 registry_report.py [/path/to/manifest.csv]   (default: ~/registry/manifest.csv)
#   python3 registry_report.py [...] [TOP_N]             (default TOP_N: 30)
# =============================================================================
import csv
import os
import sys

# Manifest columns (see include/io/artifact.hpp):
#   id,arch,horizon,cost_mult,barrier_k,conf_gate,seed,
#   holdout_pf,holdout_kelly,holdout_edge,holdout_t,holdout_neff,holdout_ci_lo,
#   oof_pf,oof_t,edge_verdict,dir
EXPECTED_COLS = [
    "id", "arch", "horizon", "cost_mult", "barrier_k", "conf_gate", "seed",
    "holdout_pf", "holdout_kelly", "holdout_edge", "holdout_t", "holdout_neff",
    "holdout_ci_lo", "oof_pf", "oof_t", "edge_verdict", "dir",
]


def _f(row, key, default=0.0):
    """Best-effort float parse; tolerant of blanks / junk in partial files."""
    try:
        v = row.get(key, "")
        if v is None or v == "":
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _i(row, key, default=0):
    try:
        v = row.get(key, "")
        if v is None or v == "":
            return default
        return int(float(v))
    except (TypeError, ValueError):
        return default


def load_rows(path):
    """Return (rows, n_lines_seen). Skips blank / header-only / malformed rows."""
    rows = []
    if not os.path.isfile(path):
        return rows
    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            if raw is None:
                continue
            # An id is mandatory; rows missing it are header dupes or junk.
            rid = (raw.get("id") or "").strip()
            if not rid or rid == "id":
                continue
            rows.append(raw)
    return rows


def is_edge_yes(row):
    return (row.get("edge_verdict") or "").strip().lower() == "yes"


def fmt(v, nd=3):
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def build_report(path, top_n):
    out = []
    w = out.append

    rows = load_rows(path)
    total = len(rows)
    edge_rows = [r for r in rows if is_edge_yes(r)]
    n_edge = len(edge_rows)

    w("=" * 78)
    w(f"  FLEET REGISTRY REPORT   manifest={path}")
    w("=" * 78)

    if total == 0:
        w("  (manifest empty or missing - no rows to report)")
        w("=" * 78)
        return "\n".join(out), rows, edge_rows

    # ----- ranked table: edge_verdict==yes by holdout_t desc, then holdout_pf desc
    ranked = sorted(
        edge_rows,
        key=lambda r: (_f(r, "holdout_t"), _f(r, "holdout_pf")),
        reverse=True,
    )

    w("")
    w(f"  TOP {min(top_n, len(ranked))} EDGE-POSITIVE MODELS  (by holdout t_stat, then holdout PF)")
    w("  " + "-" * 76)
    hdr = (f"  {'#':>3} {'arch':<5} {'horizon':>7} {'cm':>4} {'cg':>4} {'seed':>5} "
           f"{'h_t':>7} {'h_pf':>7} {'h_kelly':>8} {'h_edge':>8} {'h_neff':>7} "
           f"{'h_cilo':>8} {'oof_t':>7}")
    w(hdr)
    w("  " + "-" * 76)
    if not ranked:
        w("  (no edge_verdict==yes rows)")
    for i, r in enumerate(ranked[:top_n], 1):
        w(f"  {i:>3} {(r.get('arch') or '?'):<5} {_i(r,'horizon'):>7} "
          f"{fmt(r.get('cost_mult'),1):>4} {fmt(r.get('conf_gate'),2):>4} "
          f"{_i(r,'seed'):>5} "
          f"{fmt(r.get('holdout_t'),2):>7} {fmt(r.get('holdout_pf'),2):>7} "
          f"{fmt(r.get('holdout_kelly'),3):>8} {fmt(r.get('holdout_edge'),4):>8} "
          f"{_i(r,'holdout_neff'):>7} {fmt(r.get('holdout_ci_lo'),4):>8} "
          f"{fmt(r.get('oof_t'),2):>7}")

    # ----- summary block -----------------------------------------------------
    w("")
    w("  SUMMARY")
    w("  " + "-" * 76)
    w(f"  configs in manifest .......... {total}")
    w(f"  edge_verdict == yes .......... {n_edge}")

    if edge_rows:
        best_pf = max(edge_rows, key=lambda r: _f(r, "holdout_pf"))
        best_k = max(edge_rows, key=lambda r: _f(r, "holdout_kelly"))
        best_t = max(edge_rows, key=lambda r: _f(r, "holdout_t"))
        w(f"  best holdout PF .............. {fmt(best_pf.get('holdout_pf'),3)}  "
          f"[{best_pf.get('id','?')}]")
        w(f"  best holdout kelly .......... {fmt(best_k.get('holdout_kelly'),4)}  "
          f"[{best_k.get('id','?')}]")
        w(f"  best holdout t_stat ......... {fmt(best_t.get('holdout_t'),3)}  "
          f"[{best_t.get('id','?')}]")
    else:
        w("  best holdout PF/kelly/t ..... n/a (no edge-positive models)")

    # counts by arch (all rows + edge-yes)
    def counts_by(rows_subset, key):
        d = {}
        for r in rows_subset:
            k = (r.get(key) or "?")
            if key == "horizon":
                k = str(_i(r, "horizon"))
            d[k] = d.get(k, 0) + 1
        return d

    by_arch_all = counts_by(rows, "arch")
    by_arch_edge = counts_by(edge_rows, "arch")
    w("")
    w("  counts by arch  (edge_yes / total):")
    for k in sorted(by_arch_all):
        w(f"    {k:<6} {by_arch_edge.get(k,0):>4} / {by_arch_all.get(k):>4}")

    by_h_all = counts_by(rows, "horizon")
    by_h_edge = counts_by(edge_rows, "horizon")
    w("")
    w("  counts by horizon  (edge_yes / total):")
    for k in sorted(by_h_all, key=lambda x: int(x) if x.isdigit() else 0):
        w(f"    {k:>7} {by_h_edge.get(k,0):>4} / {by_h_all.get(k):>4}")

    w("=" * 78)
    return "\n".join(out), rows, edge_rows


def main(argv):
    # Parse: optional manifest path and optional TOP_N (in any order-ish).
    manifest = os.path.expanduser("~/registry/manifest.csv")
    top_n = 30
    pos = [a for a in argv[1:]]
    for a in pos:
        if a.isdigit():
            top_n = int(a)
        else:
            manifest = a

    report, rows, edge_rows = build_report(manifest, top_n)
    print(report)

    # Write leaderboard.txt next to the manifest (registry/leaderboard.txt).
    try:
        out_dir = os.path.dirname(os.path.abspath(manifest))
        lb = os.path.join(out_dir, "leaderboard.txt")
        with open(lb, "w") as fh:
            fh.write(report + "\n")
        print(f"\n[registry_report] wrote {lb}")
    except OSError as e:
        print(f"\n[registry_report] could not write leaderboard.txt: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
