#!/usr/bin/env python3
"""Generate a SYNTHETIC NULL XAU tick series (no real edge) in the exact 6-column
SNAPPY/PLAIN/v1 parquet the `from` reader needs. Used to build the empirical null
distribution of walk-forward holdout stats (deflated significance / PBO / FDR).

Methods:
  gbm        : geometric Brownian motion, ~zero drift (pure random walk; no structure)
  bootstrap  : block-bootstrap of REAL per-tick log-returns from a source parquet
               (preserves the return distribution + local vol clustering, DESTROYS any
                cross-time predictive structure -> a realistic null).
"""
from __future__ import annotations
import argparse, sys
import numpy as np
import pyarrow as pa, pyarrow.parquet as pq

XAU0 = 1950.0          # starting price level
TICK_MS = 250          # synthetic spacing (tick-indexed model; wall time irrelevant)


def gbm(n, seed, per_tick_vol, spread_frac):
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0, per_tick_vol, size=n)        # zero drift
    mid = XAU0 * np.exp(np.cumsum(r))
    half = 0.5 * spread_frac * mid
    ask = mid + half
    bid = mid - half
    return ask, bid, mid


def bootstrap(n, seed, src, spread_frac, block=512):
    rng = np.random.default_rng(seed)
    t = pq.read_table(src, columns=["mid"])
    m = t.column("mid").to_numpy().astype(np.float64)
    lr = np.diff(np.log(m + 1e-12))
    lr = lr[np.isfinite(lr)]
    if lr.size < block * 2:
        return gbm(n, seed, 1e-5, spread_frac)
    # stitch random blocks of real returns until we have n-1 of them
    nb = (n // block) + 2
    starts = rng.integers(0, lr.size - block, size=nb)
    seq = np.concatenate([lr[s:s + block] for s in starts])[: n - 1]
    mid = XAU0 * np.exp(np.cumsum(np.concatenate([[0.0], seq])))
    half = 0.5 * spread_frac * mid
    return mid + half, mid - half, mid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--method", choices=["gbm", "bootstrap"], default="gbm")
    ap.add_argument("--n-ticks", type=int, default=2_000_000)
    ap.add_argument("--src", default="")                 # for bootstrap
    ap.add_argument("--per-tick-vol", type=float, default=1.2e-5)
    ap.add_argument("--spread-frac", type=float, default=1.0e-4)  # ~0.0001 of price
    args = ap.parse_args()

    n = args.n_ticks
    if args.method == "bootstrap" and args.src:
        ask, bid, mid = bootstrap(n, args.seed, args.src, args.spread_frac)
    else:
        ask, bid, mid = gbm(n, args.seed, args.per_tick_vol, args.spread_frac)
    n = mid.size
    rng = np.random.default_rng(args.seed + 9991)
    av = rng.uniform(0.5, 3.0, size=n).astype(np.float32)
    bv = rng.uniform(0.5, 3.0, size=n).astype(np.float32)
    t = (np.arange(n, dtype=np.int64) * TICK_MS) + 1_600_000_000_000

    table = pa.table({
        "ask": pa.array(ask, pa.float64()), "bid": pa.array(bid, pa.float64()),
        "mid": pa.array(mid, pa.float64()),
        "ask_vol": pa.array(av, pa.float32()), "bid_vol": pa.array(bv, pa.float32()),
        "time": pa.array(t, pa.int64()),
    })
    pq.write_table(table, args.out, compression="snappy", use_dictionary=False,
                   data_page_version="1.0", version="1.0", row_group_size=1_048_576)
    print(f"WROTE {args.out} method={args.method} seed={args.seed} rows={n:,}")


if __name__ == "__main__":
    sys.exit(main())
