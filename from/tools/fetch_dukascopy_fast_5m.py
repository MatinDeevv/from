#!/usr/bin/env python3
"""Fast one-pass Dukascopy bi5 downloader to 5m parquet."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import lzma
import time
import urllib.error
import urllib.request
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


HOUR_MS = 3_600_000
TF_MS = 300_000
DTYPE = np.dtype([("offset", ">u4"), ("ask", ">u4"), ("bid", ">u4"), ("ask_vol", ">f4"), ("bid_vol", ">f4")])


def parse_time(value: str) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def url(instrument: str, hour_ms: int) -> str:
    d = dt.datetime.fromtimestamp(hour_ms / 1000, tz=dt.timezone.utc)
    return f"https://datafeed.dukascopy.com/datafeed/{instrument}/{d.year}/{d.month - 1:02d}/{d.day:02d}/{d.hour:02d}h_ticks.bi5"


def fetch_hour(instrument: str, hour_ms: int, timeout: int) -> tuple[int, bytes | None, str | None, int | None]:
    try:
        req = urllib.request.Request(url(instrument, hour_ms), headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            status = getattr(response, "status", 200)
            body = response.read()
        payload = lzma.decompress(body)
        if not payload or len(payload) % DTYPE.itemsize:
            return hour_ms, None, f"bad_payload_{len(payload)}", status
        return hour_ms, payload, None, status
    except urllib.error.HTTPError as exc:
        return hour_ms, None, f"HTTP_{exc.code}", exc.code
    except Exception as exc:
        return hour_ms, None, type(exc).__name__, None


def aggregate(hour_ms: int, payload: bytes, start_ms: int, end_ms: int, scale: float) -> pd.DataFrame:
    r = np.frombuffer(payload, dtype=DTYPE)
    actual = hour_ms + r["offset"].astype(np.int64)
    keep = (actual >= start_ms) & (actual < end_ms)
    if not keep.any():
        return pd.DataFrame()
    actual = actual[keep]
    ask = r["ask"][keep].astype(np.float64) / scale
    bid = r["bid"][keep].astype(np.float64) / scale
    mid = (ask + bid) * 0.5
    spread = ask - bid
    bucket = (actual // TF_MS) * TF_MS
    df = pd.DataFrame({"bucket_ms": bucket, "tick_ms": actual, "mid": mid, "spread": spread, "ask_vol": r["ask_vol"][keep].astype(np.float64), "bid_vol": r["bid_vol"][keep].astype(np.float64)})
    return df.groupby("bucket_ms", sort=True).agg(
        open=("mid", "first"),
        high=("mid", "max"),
        low=("mid", "min"),
        close=("mid", "last"),
        spread=("spread", "mean"),
        ticks=("mid", "size"),
        ask_vol=("ask_vol", "mean"),
        bid_vol=("bid_vol", "mean"),
        first_tick_ms=("tick_ms", "first"),
        last_tick_ms=("tick_ms", "last"),
    ).reset_index()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--instrument", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--price-scale", type=float, required=True)
    p.add_argument("--workers", type=int, default=64)
    p.add_argument("--timeout", type=int, default=8)
    args = p.parse_args()

    start = parse_time(args.start)
    end = parse_time(args.end)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    first = (start_ms // HOUR_MS) * HOUR_MS
    last = ((end_ms - 1) // HOUR_MS) * HOUR_MS
    hours = list(range(first, last + HOUR_MS, HOUR_MS))
    frames = []
    status_counts: dict[str, int] = {}
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_hour, args.instrument.upper(), h, args.timeout): h for h in hours}
        for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            h, payload, err, status = fut.result()
            key = "ok" if payload is not None else (err or "unknown")
            status_counts[key] = status_counts.get(key, 0) + 1
            if payload is not None:
                frame = aggregate(h, payload, start_ms, end_ms, args.price_scale)
                if not frame.empty:
                    frames.append(frame)
            if i % 1000 == 0:
                print(f"{args.instrument} {i}/{len(hours)} frames={len(frames)} statuses={status_counts}")
    if frames:
        bars = pd.concat(frames, ignore_index=True)
        bars = bars.groupby("bucket_ms", sort=True).agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            spread=("spread", "mean"),
            ticks=("ticks", "sum"),
            ask_vol=("ask_vol", "mean"),
            bid_vol=("bid_vol", "mean"),
            first_tick_ms=("first_tick_ms", "min"),
            last_tick_ms=("last_tick_ms", "max"),
        ).reset_index()
        bars["time"] = pd.to_datetime(bars["bucket_ms"], unit="ms", utc=True)
        bars = bars[["time", "bucket_ms", "open", "high", "low", "close", "spread", "ticks", "ask_vol", "bid_vol", "first_tick_ms", "last_tick_ms"]]
    else:
        bars = pd.DataFrame(columns=["time", "bucket_ms", "open", "high", "low", "close", "spread", "ticks", "ask_vol", "bid_vol", "first_tick_ms", "last_tick_ms"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.register("bars", bars)
    con.execute(f"COPY bars TO '{str(out).replace(chr(39), chr(39)*2)}' (FORMAT PARQUET)")
    report = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "instrument": args.instrument.upper(),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "price_scale": args.price_scale,
        "hours_requested": len(hours),
        "status_counts": status_counts,
        "bars": int(len(bars)),
        "first_bar": str(bars["time"].min()) if len(bars) else None,
        "last_bar": str(bars["time"].max()) if len(bars) else None,
        "elapsed_seconds": time.time() - t0,
    }
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
