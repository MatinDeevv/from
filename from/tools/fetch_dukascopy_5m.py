#!/usr/bin/env python3
"""Fetch Dukascopy ticks for one instrument and aggregate to 5-minute bars."""

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

import numpy as np
import pandas as pd
import duckdb


HOUR_MS = 3_600_000
TF_MS = 5 * 60_000
RECORD_DTYPE = np.dtype(
    [
        ("offset", ">u4"),
        ("ask", ">u4"),
        ("bid", ">u4"),
        ("ask_vol", ">f4"),
        ("bid_vol", ">f4"),
    ]
)


def parse_time(value: str) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def archive_url(instrument: str, hour_ms: int) -> str:
    date = dt.datetime.fromtimestamp(hour_ms / 1000, tz=dt.timezone.utc)
    return (
        f"https://datafeed.dukascopy.com/datafeed/{instrument}/"
        f"{date.year}/{date.month - 1:02d}/{date.day:02d}/{date.hour:02d}h_ticks.bi5"
    )


def fetch_hour(instrument: str, hour_ms: int) -> tuple[int, bytes | None, str | None]:
    url = archive_url(instrument, hour_ms)
    backoffs_503 = [2, 4, 8, 16, 32]
    attempt = 0
    while True:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as response:
                content = response.read()
            time.sleep(0.5)
            if not content:
                return hour_ms, None, "empty"
            payload = lzma.decompress(content)
            if len(payload) == 0 or len(payload) % RECORD_DTYPE.itemsize != 0:
                return hour_ms, None, f"bad_payload_{len(payload)}"
            return hour_ms, payload, None
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return hour_ms, None, "404"
            if exc.code == 503:
                if attempt < len(backoffs_503):
                    delay = backoffs_503[attempt]
                    print(f"warning: 503 for {instrument} {hour_ms}; retry {attempt + 1}/5 after {delay}s", flush=True)
                    time.sleep(delay)
                    attempt += 1
                    continue
                print(f"warning: skipping {instrument} {hour_ms} after 5 retries on 503", flush=True)
                return hour_ms, None, "503_after_5_retries"
            text = repr(exc)
            if attempt < 3:
                time.sleep(0.5 * (attempt + 1))
                attempt += 1
                continue
            return hour_ms, None, text
        except Exception as exc:
            text = repr(exc)
            if attempt < 3:
                time.sleep(0.5 * (attempt + 1))
                attempt += 1
                continue
            return hour_ms, None, text


def aggregate_payload(hour_ms: int, payload: bytes, start_ms: int, end_ms: int, price_scale: float) -> pd.DataFrame:
    records = np.frombuffer(payload, dtype=RECORD_DTYPE)
    actual_ms = hour_ms + records["offset"].astype(np.int64)
    keep = (actual_ms >= start_ms) & (actual_ms < end_ms)
    if not keep.any():
        return pd.DataFrame()
    actual_ms = actual_ms[keep]
    ask = records["ask"][keep].astype(np.float64) / price_scale
    bid = records["bid"][keep].astype(np.float64) / price_scale
    mid = (ask + bid) * 0.5
    spread = ask - bid
    bucket = (actual_ms // TF_MS) * TF_MS
    df = pd.DataFrame(
        {
            "bucket_ms": bucket,
            "tick_ms": actual_ms,
            "mid": mid,
            "spread": spread,
            "ask_vol": records["ask_vol"][keep].astype(np.float64),
            "bid_vol": records["bid_vol"][keep].astype(np.float64),
        }
    )
    grouped = df.groupby("bucket_ms", sort=True)
    return grouped.agg(
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


def default_scale(instrument: str) -> float:
    if instrument in {"XAUUSD", "XAGUSD"}:
        return 1000.0
    if instrument.endswith("JPY"):
        return 1000.0
    return 100000.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instrument", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--price-scale", type=float, default=None)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    instrument = args.instrument.upper()
    price_scale = float(args.price_scale or default_scale(instrument))
    start = parse_time(args.start)
    end = parse_time(args.end)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    first_hour = (start_ms // HOUR_MS) * HOUR_MS
    last_hour = ((end_ms - 1) // HOUR_MS) * HOUR_MS
    hours = list(range(first_hour, last_hour + HOUR_MS, HOUR_MS))

    frames = []
    errors = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(fetch_hour, instrument, hour): hour for hour in hours}
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            hour_ms, payload, err = fut.result()
            if payload is not None:
                frame = aggregate_payload(hour_ms, payload, start_ms, end_ms, price_scale)
                if not frame.empty:
                    frames.append(frame)
            elif err not in {"404", "empty"}:
                errors[str(hour_ms)] = err
            if i % 250 == 0:
                print(f"fetched {i}/{len(hours)} hours frames={len(frames)} errors={len(errors)}")

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
    if out.suffix.lower() == ".parquet":
        con = duckdb.connect()
        con.register("bars", bars)
        con.execute(f"COPY bars TO '{str(out).replace(chr(39), chr(39) * 2)}' (FORMAT PARQUET)")
    else:
        bars.to_csv(out, index=False)
    report = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "instrument": instrument,
        "price_scale": price_scale,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "hours_requested": len(hours),
        "bars": int(len(bars)),
        "first_bar": str(bars["time"].min()) if len(bars) else None,
        "last_bar": str(bars["time"].max()) if len(bars) else None,
        "errors": errors,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
