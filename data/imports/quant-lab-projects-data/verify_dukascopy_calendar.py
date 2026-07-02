#!/usr/bin/env python3
"""Verify and resolve modulo timestamp cycles against Dukascopy's tick archive."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import lzma
import struct
import threading
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests


UINT32_MS = 2**32
HOUR_MS = 3_600_000
RECORD_SIZE = 20
MIN_CYCLE = 240
MAX_CYCLE = 414
THREADS = 32

_thread_state = threading.local()


def session() -> requests.Session:
    if not hasattr(_thread_state, "session"):
        _thread_state.session = requests.Session()
    return _thread_state.session


def archive_url(epoch_ms: int) -> str:
    date = datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc)
    return (
        "https://datafeed.dukascopy.com/datafeed/XAUUSD/"
        f"{date.year}/{date.month - 1:02d}/{date.day:02d}/{date.hour:02d}h_ticks.bi5"
    )


def expected_record(
    epoch_ms: int, ask: float, bid: float, ask_vol: float, bid_vol: float
) -> bytes:
    return struct.pack(
        ">3I2f",
        epoch_ms % HOUR_MS,
        round(ask * 1000),
        round(bid * 1000),
        ask_vol,
        bid_vol,
    )


def fetch(url: str) -> tuple[str, bytes | None]:
    try:
        response = session().get(url, timeout=30)
        if response.status_code != 200 or not response.content:
            return url, None
        payload = lzma.decompress(response.content)
        if len(payload) % RECORD_SIZE:
            return url, None
        return url, payload
    except Exception:
        return url, None


def fetch_all(urls: set[str]) -> dict[str, bytes | None]:
    result: dict[str, bytes | None] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as executor:
        for index, (url, payload) in enumerate(executor.map(fetch, sorted(urls)), 1):
            result[url] = payload
            if index % 500 == 0:
                print(f"downloaded {index:,}/{len(urls):,} archive hours", flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("mapping", type=Path)
    parser.add_argument("--output", type=Path, default=Path("dated/dukascopy_verification.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parquet = pq.ParquetFile(args.input)
    mapping = pd.read_csv(args.mapping)

    fingerprints = []
    selected_requests: dict[str, list[tuple[int, int, bytes]]] = {}
    for group_id in range(parquet.num_row_groups):
        table = parquet.read_row_group(
            group_id, columns=["time", "ask", "bid", "ask_vol", "bid_vol"]
        )
        raw_ms = int(table["time"][0].cast(pa.int64()).as_py())
        values = [table[name][0].as_py() for name in ("ask", "bid", "ask_vol", "bid_vol")]
        cycle = int(mapping.loc[group_id, "timestamp_cycle"])
        epoch_ms = raw_ms + cycle * UINT32_MS
        record = expected_record(epoch_ms, *values)
        url = archive_url(epoch_ms)
        fingerprints.append((raw_ms, values, record))
        selected_requests.setdefault(url, []).append((group_id, cycle, record))

    print(f"testing {len(selected_requests):,} currently selected archive hours", flush=True)
    selected_payloads = fetch_all(set(selected_requests))
    resolved: dict[int, int] = {}
    selected_archive_available: dict[int, bool] = {}
    for url, checks in selected_requests.items():
        payload = selected_payloads[url]
        for group_id, cycle, record in checks:
            selected_archive_available[group_id] = payload is not None
            if payload is not None and record in payload:
                resolved[group_id] = cycle

    unresolved = [group_id for group_id in range(parquet.num_row_groups) if group_id not in resolved]
    print(
        f"selected-cycle exact matches: {len(resolved):,}/{parquet.num_row_groups:,}; "
        f"searching all cycles for {len(unresolved):,} unresolved groups",
        flush=True,
    )

    search_requests: dict[str, list[tuple[int, int, bytes]]] = {}
    for group_id in unresolved:
        raw_ms, values, _ = fingerprints[group_id]
        for cycle in range(MIN_CYCLE, MAX_CYCLE + 1):
            epoch_ms = raw_ms + cycle * UINT32_MS
            record = expected_record(epoch_ms, *values)
            url = archive_url(epoch_ms)
            search_requests.setdefault(url, []).append((group_id, cycle, record))

    search_payloads = fetch_all(set(search_requests))
    matches: dict[int, list[int]] = {group_id: [] for group_id in unresolved}
    for url, checks in search_requests.items():
        payload = search_payloads[url]
        if payload is None:
            continue
        for group_id, cycle, record in checks:
            if record in payload:
                matches[group_id].append(cycle)

    rows = []
    for group_id in range(parquet.num_row_groups):
        inferred_cycle = int(mapping.loc[group_id, "timestamp_cycle"])
        exact_cycles = [resolved[group_id]] if group_id in resolved else matches.get(group_id, [])
        verified_cycle = exact_cycles[0] if len(exact_cycles) == 1 else None
        raw_ms = fingerprints[group_id][0]
        rows.append(
            {
                "source_row_group": group_id,
                "inferred_cycle": inferred_cycle,
                "inferred_cycle_exact_match": inferred_cycle in exact_cycles,
                "selected_archive_available": selected_archive_available.get(group_id, False),
                "exact_match_count": len(exact_cycles),
                "verified_cycle": verified_cycle,
                "verified_start_utc": (
                    pd.to_datetime(raw_ms + verified_cycle * UINT32_MS, unit="ms", utc=True)
                    if verified_cycle is not None
                    else pd.NaT
                ),
            }
        )

    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    summary = {
        "source_row_groups": parquet.num_row_groups,
        "inferred_cycles_exactly_confirmed": int(output.inferred_cycle_exact_match.sum()),
        "uniquely_verified_cycles": int(output.verified_cycle.notna().sum()),
        "unresolved_cycles": int(output.verified_cycle.isna().sum()),
        "inferred_cycles_corrected": int(
            (
                output.verified_cycle.notna()
                & (output.verified_cycle != output.inferred_cycle)
            ).sum()
        ),
        "output": str(args.output.resolve()),
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
