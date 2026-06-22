#!/usr/bin/env python3
"""Download Dukascopy tick archives and write RAW TICKS as the 6-column parquet the
`from` engine's hand-rolled reader expects.

Reader contract (src/data/parquet_reader.cpp): exactly six columns, in THIS order —
  ask(double), bid(double), mid(double), ask_vol(float), bid_vol(float), time(int64 ms)
SNAPPY compression, DATA_PAGE v1, PLAIN encoding (use_dictionary off), nullable columns
(so definition levels are emitted). Rows must be in chronological order.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import lzma
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

HOUR_MS = 3_600_000
DTYPE = np.dtype([("offset", ">u4"), ("ask", ">u4"), ("bid", ">u4"),
                  ("ask_vol", ">f4"), ("bid_vol", ">f4")])


def parse_time(value: str) -> dt.datetime:
    p = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if p.tzinfo is None:
        p = p.replace(tzinfo=dt.timezone.utc)
    return p.astimezone(dt.timezone.utc)


def url(instrument: str, hour_ms: int) -> str:
    d = dt.datetime.fromtimestamp(hour_ms / 1000, tz=dt.timezone.utc)
    return (f"https://datafeed.dukascopy.com/datafeed/{instrument}/"
            f"{d.year}/{d.month - 1:02d}/{d.day:02d}/{d.hour:02d}h_ticks.bi5")


def fetch_hour(instrument: str, hour_ms: int, timeout: int, retries: int = 6):
    # Dukascopy 503s aggressively under load — retry transient errors with
    # exponential backoff + jitter; 404 is a genuine no-data (closed) hour.
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url(instrument, hour_ms),
                                         headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read()
            if not body:
                return hour_ms, None, "empty"
            payload = lzma.decompress(body)
            if not payload or len(payload) % DTYPE.itemsize:
                return hour_ms, None, f"bad_{len(payload)}"
            return hour_ms, payload, None
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return hour_ms, None, "404"  # market closed hour; expected
            if attempt == retries:
                return hour_ms, None, f"HTTP_{e.code}"
        except Exception as e:
            if attempt == retries:
                return hour_ms, None, type(e).__name__
        # exp backoff with jitter; capped so a few rate-limited hours don't gate wall time
        time.sleep(min(8.0, 0.5 * (2 ** attempt)) * (0.5 + 0.7 * random.random()))
    return hour_ms, None, "exhausted"


def decode(hour_ms: int, payload: bytes, scale: float):
    r = np.frombuffer(payload, dtype=DTYPE)
    if r.size == 0:
        return None
    t = hour_ms + r["offset"].astype(np.int64)
    ask = r["ask"].astype(np.float64) / scale
    bid = r["bid"].astype(np.float64) / scale
    mid = (ask + bid) * 0.5
    return (t, ask, bid, mid,
            r["ask_vol"].astype(np.float32), r["bid_vol"].astype(np.float32))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instrument", default="XAUUSD")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--price-scale", type=float, default=1000.0)  # XAUUSD: 3 decimals
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--timeout", type=int, default=15)
    ap.add_argument("--row-group", type=int, default=1_048_576)
    args = ap.parse_args()

    start_ms = int(parse_time(args.start).timestamp() * 1000)
    end_ms = int(parse_time(args.end).timestamp() * 1000)
    first = (start_ms // HOUR_MS) * HOUR_MS
    last = ((end_ms - 1) // HOUR_MS) * HOUR_MS
    hours = list(range(first, last + HOUR_MS, HOUR_MS))

    parts: list[tuple] = []
    statuses: dict[str, int] = {}
    n_ticks = 0
    t0 = time.time()
    inst = args.instrument.upper()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_hour, inst, h, args.timeout): h for h in hours}
        for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            h, payload, err = fut.result()
            key = "ok" if payload is not None else err
            statuses[key] = statuses.get(key, 0) + 1
            if payload is not None:
                d = decode(h, payload, args.price_scale)
                if d is not None:
                    parts.append(d)
                    n_ticks += d[0].size
            if i % 2000 == 0:
                rate = i / max(1e-9, time.time() - t0)
                print(f"{inst} {i}/{len(hours)} hrs  ticks={n_ticks:,}  "
                      f"{rate:.0f} hr/s  statuses={statuses}", flush=True)

    if not parts:
        print(f"NO DATA. statuses={statuses}")
        return 1

    print(f"concatenating {len(parts):,} hourly blocks, {n_ticks:,} ticks ...", flush=True)
    t = np.concatenate([p[0] for p in parts])
    ask = np.concatenate([p[1] for p in parts])
    bid = np.concatenate([p[2] for p in parts])
    mid = np.concatenate([p[3] for p in parts])
    av = np.concatenate([p[4] for p in parts])
    bv = np.concatenate([p[5] for p in parts])
    del parts

    print("sorting chronologically ...", flush=True)
    order = np.argsort(t, kind="stable")
    t, ask, bid, mid, av, bv = t[order], ask[order], bid[order], mid[order], av[order], bv[order]
    keep = (t >= start_ms) & (t < end_ms)
    t, ask, bid, mid, av, bv = t[keep], ask[keep], bid[keep], mid[keep], av[keep], bv[keep]

    # Column order/types MUST match the reader exactly.
    table = pa.table({
        "ask": pa.array(ask, type=pa.float64()),
        "bid": pa.array(bid, type=pa.float64()),
        "mid": pa.array(mid, type=pa.float64()),
        "ask_vol": pa.array(av, type=pa.float32()),
        "bid_vol": pa.array(bv, type=pa.float32()),
        "time": pa.array(t, type=pa.int64()),
    })
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(out), compression="snappy", use_dictionary=False,
                   data_page_version="1.0", version="1.0",
                   row_group_size=args.row_group)
    sz = out.stat().st_size
    print(f"WROTE {out}  rows={table.num_rows:,}  bytes={sz:,} ({sz/1e9:.2f} GB)  "
          f"span=[{t[0]}..{t[-1]}]  elapsed={time.time()-t0:.0f}s  statuses={statuses}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
