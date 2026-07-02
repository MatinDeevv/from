#!/usr/bin/env python3
"""Fill missing MASTER_DATASET minutes using real Dukascopy XAUUSD ticks."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import lzma
import struct
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import requests


HOUR_MS = 3_600_000
MINUTE_MS = 60_000
RECORD_SIZE = 20
THREADS = 32
_thread_state = threading.local()

FILL_SCHEMA = pa.schema(
    [
        ("timestamp_utc", pa.timestamp("us", tz="UTC")),
        ("first_tick_time_utc", pa.timestamp("us", tz="UTC")),
        ("last_tick_time_utc", pa.timestamp("us", tz="UTC")),
        ("max_source_tick_gap_ms", pa.int64()),
        ("tick_count", pa.int64()),
        ("source_group_count", pa.int32()),
        ("timestamp_mapping_score_max", pa.float32()),
        ("timestamp_mapping_low_confidence", pa.bool_()),
        ("timestamp_dukascopy_exact_verified_all", pa.bool_()),
        ("timestamp_dukascopy_exact_verified_any", pa.bool_()),
        ("timestamp_dukascopy_cycle_corrected_any", pa.bool_()),
        ("external_gap_fill", pa.bool_()),
        *[
            (f"{name}_{field}", pa.float64())
            for name in ("bid", "ask", "mid", "spread")
            for field in ("open", "high", "low", "close")
        ],
        ("spread_mean", pa.float64()),
        ("bid_vol_mean", pa.float64()),
        ("ask_vol_mean", pa.float64()),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("master", type=Path, default=Path("MASTER_DATASET.parquet"))
    parser.add_argument("--output", type=Path, default=Path("MASTER_DATASET_FILLED.parquet"))
    parser.add_argument("--fills", type=Path, default=Path("external_gap_fills.parquet"))
    parser.add_argument("--report", type=Path, default=Path("master_gap_fill_report.json"))
    parser.add_argument("--threads", type=int, default=THREADS)
    parser.add_argument("--core-weekday-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def session() -> requests.Session:
    if not hasattr(_thread_state, "session"):
        _thread_state.session = requests.Session()
    return _thread_state.session


def archive_url(hour_ms: int) -> str:
    date = datetime.fromtimestamp(hour_ms / 1000, tz=timezone.utc)
    return (
        "https://datafeed.dukascopy.com/datafeed/XAUUSD/"
        f"{date.year}/{date.month - 1:02d}/{date.day:02d}/{date.hour:02d}h_ticks.bi5"
    )


def fetch_hour(item: tuple[int, list[tuple[int, int]]]) -> tuple[int, bytes | None]:
    hour_ms, _ = item
    for attempt in range(5):
        try:
            response = session().get(archive_url(hour_ms), timeout=30)
            if response.status_code == 404 or not response.content:
                return hour_ms, None
            if response.status_code != 200:
                retry_after = response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else 0.5 * (attempt + 1)
                time.sleep(min(delay, 10))
                continue
            payload = lzma.decompress(response.content)
            if not payload or len(payload) % RECORD_SIZE:
                return hour_ms, None
            return hour_ms, payload
        except Exception:
            if attempt < 4:
                time.sleep(0.5 * (attempt + 1))
    return hour_ms, None


def missing_intervals(
    master: Path, core_weekday_only: bool = False
) -> tuple[list[tuple[int, int]], dict[int, list[tuple[int, int]]]]:
    connection = duckdb.connect()
    source = str(master).replace("\\", "/").replace("'", "''")
    core_filter = ""
    if core_weekday_only:
        core_filter = """
          AND cast(epoch_ms(previous_bar_ms) AS DATE)
            = cast(epoch_ms(next_bar_ms) AS DATE)
          AND extract(isodow FROM epoch_ms(next_bar_ms)) BETWEEN 1 AND 5
          AND extract(hour FROM epoch_ms(previous_bar_ms)) < 20
          AND extract(hour FROM epoch_ms(next_bar_ms)) < 20
        """
    rows = connection.execute(
        f"""
        WITH x AS (
          SELECT
            epoch_ms(timestamp_utc) AS next_bar_ms,
            lag(epoch_ms(timestamp_utc)) OVER (ORDER BY timestamp_utc) AS previous_bar_ms
          FROM read_parquet('{source}')
        )
        SELECT previous_bar_ms + {MINUTE_MS} AS start_ms, next_bar_ms AS end_ms
        FROM x
        WHERE next_bar_ms > previous_bar_ms + {MINUTE_MS}
        {core_filter}
        ORDER BY start_ms
        """
    ).fetchall()
    connection.close()

    intervals = [(int(start), int(end)) for start, end in rows]
    hours: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for start, end in intervals:
        hour = start // HOUR_MS * HOUR_MS
        last_hour = (end - 1) // HOUR_MS * HOUR_MS
        while hour <= last_hour:
            date = datetime.fromtimestamp(hour / 1000, tz=timezone.utc)
            # XAUUSD is normally closed Saturday and most of Sunday.
            if date.weekday() < 5 or (date.weekday() == 6 and date.hour >= 20):
                hours[hour].append((start, end))
            hour += HOUR_MS
    return intervals, hours


def aggregate_payload(
    hour_ms: int, payload: bytes, intervals: list[tuple[int, int]]
) -> pa.Table | None:
    records = np.frombuffer(
        payload,
        dtype=np.dtype(
            [
                ("offset", ">u4"),
                ("ask", ">u4"),
                ("bid", ">u4"),
                ("ask_vol", ">f4"),
                ("bid_vol", ">f4"),
            ]
        ),
    )
    actual_ms = hour_ms + records["offset"].astype(np.int64)
    keep = np.zeros(len(records), dtype=bool)
    for start, end in intervals:
        keep |= (actual_ms >= start) & (actual_ms < end)
    if not keep.any():
        return None

    actual_ms = actual_ms[keep]
    ask = records["ask"][keep].astype(np.float64) / 1000
    bid = records["bid"][keep].astype(np.float64) / 1000
    ask_vol = records["ask_vol"][keep].astype(np.float64)
    bid_vol = records["bid_vol"][keep].astype(np.float64)
    mid = (ask + bid) / 2
    spread = ask - bid
    bars = actual_ms // MINUTE_MS
    starts = np.r_[0, np.flatnonzero(bars[1:] != bars[:-1]) + 1]
    ends = np.r_[starts[1:], len(bars)]
    counts = ends - starts
    gaps = np.diff(actual_ms, prepend=actual_ms[0])
    bar_ms = bars[starts] * MINUTE_MS

    data: dict[str, object] = {
        "timestamp_utc": pa.array(bar_ms * 1000, type=pa.timestamp("us", tz="UTC")),
        "first_tick_time_utc": pa.array(
            actual_ms[starts] * 1000, type=pa.timestamp("us", tz="UTC")
        ),
        "last_tick_time_utc": pa.array(
            actual_ms[ends - 1] * 1000, type=pa.timestamp("us", tz="UTC")
        ),
        "max_source_tick_gap_ms": np.maximum.reduceat(gaps, starts).astype(np.int64),
        "tick_count": counts.astype(np.int64),
        "source_group_count": np.ones(len(starts), dtype=np.int32),
        "timestamp_mapping_score_max": np.zeros(len(starts), dtype=np.float32),
        "timestamp_mapping_low_confidence": np.zeros(len(starts), dtype=bool),
        "timestamp_dukascopy_exact_verified_all": np.ones(len(starts), dtype=bool),
        "timestamp_dukascopy_exact_verified_any": np.ones(len(starts), dtype=bool),
        "timestamp_dukascopy_cycle_corrected_any": np.zeros(len(starts), dtype=bool),
        "external_gap_fill": np.ones(len(starts), dtype=bool),
    }
    for name, values in (("bid", bid), ("ask", ask), ("mid", mid), ("spread", spread)):
        data[f"{name}_open"] = values[starts]
        data[f"{name}_high"] = np.maximum.reduceat(values, starts)
        data[f"{name}_low"] = np.minimum.reduceat(values, starts)
        data[f"{name}_close"] = values[ends - 1]
    data["spread_mean"] = np.add.reduceat(spread, starts) / counts
    data["bid_vol_mean"] = np.add.reduceat(bid_vol, starts) / counts
    data["ask_vol_mean"] = np.add.reduceat(ask_vol, starts) / counts
    return pa.Table.from_pydict(data, schema=FILL_SCHEMA)


def download_fills(
    hours: dict[int, list[tuple[int, int]]], path: Path, threads: int
) -> dict[str, int]:
    writer = pq.ParquetWriter(path, FILL_SCHEMA, compression="zstd", compression_level=6)
    available = 0
    filled_hours = 0
    fill_bars = 0
    fill_ticks = 0
    items = sorted(hours.items())
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        for index, (hour_ms, payload) in enumerate(executor.map(fetch_hour, items), 1):
            if payload is not None:
                available += 1
                table = aggregate_payload(hour_ms, payload, hours[hour_ms])
                if table is not None:
                    writer.write_table(table)
                    filled_hours += 1
                    fill_bars += table.num_rows
                    fill_ticks += int(pc.sum(table["tick_count"]).as_py())
            if index % 1000 == 0:
                print(
                    f"archive hours {index:,}/{len(items):,}; "
                    f"filled bars {fill_bars:,}",
                    flush=True,
                )
    writer.close()
    return {
        "requested_archive_hours": len(items),
        "available_archive_hours": available,
        "archive_hours_with_fill_ticks": filled_hours,
        "candidate_fill_bars_before_deduplication": fill_bars,
        "candidate_fill_ticks_before_deduplication": fill_ticks,
    }


def merge(master: Path, fills: Path, output: Path) -> None:
    connection = duckdb.connect()
    source = str(master).replace("\\", "/").replace("'", "''")
    fill_source = str(fills).replace("\\", "/").replace("'", "''")
    destination = str(output).replace("\\", "/").replace("'", "''")
    connection.execute("SET preserve_insertion_order=false")
    source_columns = {
        row[0]
        for row in connection.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{source}')"
        ).fetchall()
    }
    if "external_gap_fill" in source_columns:
        master_select = "* EXCLUDE (gap_before_ms, missing_time_before_ms)"
    else:
        master_select = (
            "* EXCLUDE (gap_before_ms, missing_time_before_ms), "
            "false AS external_gap_fill"
        )
    connection.execute(
        f"""
        COPY (
          WITH combined AS (
            SELECT
              {master_select}
            FROM read_parquet('{source}')
            UNION ALL BY NAME
            SELECT * FROM read_parquet('{fill_source}')
          ),
          deduplicated AS (
            SELECT * EXCLUDE (priority)
            FROM (
              SELECT *, row_number() OVER (
                PARTITION BY timestamp_utc
                ORDER BY external_gap_fill ASC
              ) AS priority
              FROM combined
            )
            WHERE priority = 1
          ),
          with_previous AS (
            SELECT *,
              lag(last_tick_time_utc) OVER (ORDER BY timestamp_utc)
                AS previous_last_tick_time_utc,
              lag(timestamp_utc) OVER (ORDER BY timestamp_utc)
                AS previous_timestamp_utc
            FROM deduplicated
          )
          SELECT
            timestamp_utc,
            first_tick_time_utc,
            last_tick_time_utc,
            coalesce(date_diff(
              'millisecond', previous_last_tick_time_utc, first_tick_time_utc
            ), 0)::BIGINT AS gap_before_ms,
            greatest(coalesce(date_diff(
              'millisecond', previous_timestamp_utc, timestamp_utc
            ) - {MINUTE_MS}, 0), 0)::BIGINT AS missing_time_before_ms,
            * EXCLUDE (
              timestamp_utc, first_tick_time_utc, last_tick_time_utc,
              previous_last_tick_time_utc, previous_timestamp_utc
            )
          FROM with_previous
          ORDER BY timestamp_utc
        ) TO '{destination}'
        (FORMAT PARQUET, COMPRESSION ZSTD, COMPRESSION_LEVEL 6, ROW_GROUP_SIZE 200000)
        """
    )
    connection.close()


def validate(path: Path) -> dict[str, object]:
    connection = duckdb.connect()
    source = str(path).replace("\\", "/").replace("'", "''")
    result = connection.execute(
        f"""
        WITH x AS (
          SELECT *, lag(timestamp_utc) OVER (ORDER BY timestamp_utc) AS previous
          FROM read_parquet('{source}')
        )
        SELECT
          count(*) AS bars,
          sum(tick_count) AS ticks,
          count(*) FILTER (WHERE external_gap_fill) AS external_gap_fill_bars,
          count(*) FILTER (WHERE missing_time_before_ms > 0) AS remaining_gap_events,
          sum(missing_time_before_ms) / {MINUTE_MS} AS remaining_missing_minutes,
          count(*) FILTER (
            WHERE missing_time_before_ms >= 3600000
              AND cast(timestamp_utc AT TIME ZONE 'UTC' AS DATE)
                = cast(previous AT TIME ZONE 'UTC' AS DATE)
              AND extract(isodow FROM timestamp_utc AT TIME ZONE 'UTC') BETWEEN 1 AND 5
              AND extract(hour FROM previous AT TIME ZONE 'UTC') < 20
              AND extract(hour FROM timestamp_utc AT TIME ZONE 'UTC') < 20
          ) AS remaining_core_weekday_gaps_ge_1h,
          count(*) FILTER (WHERE timestamp_utc <= previous) AS non_increasing,
          count(*) FILTER (WHERE gap_before_ms < 0) AS negative_gap,
          count(*) FILTER (
            WHERE ask_open < bid_open OR ask_close < bid_close OR spread_low < 0
          ) AS bad_spread
        FROM x
        """
    ).fetchone()
    connection.close()
    names = [
        "bars",
        "ticks",
        "external_gap_fill_bars",
        "remaining_gap_events",
        "remaining_missing_minutes",
        "remaining_core_weekday_gaps_ge_1h",
        "non_increasing",
        "negative_gap",
        "bad_spread",
    ]
    return dict(zip(names, result))


def main() -> None:
    args = parse_args()
    for path in (args.output, args.fills, args.report):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
        if path.exists():
            path.unlink()

    started = time.time()
    intervals, hours = missing_intervals(args.master, args.core_weekday_only)
    print(
        f"missing intervals: {len(intervals):,}; archive hours to check: {len(hours):,}",
        flush=True,
    )
    download_report = download_fills(hours, args.fills, args.threads)
    merge(args.master, args.fills, args.output)
    report = {
        "input": str(args.master.resolve()),
        "output": str(args.output.resolve()),
        "fill_source": "Dukascopy XAUUSD hourly .bi5 tick archive",
        "fill_policy": (
            "Only real Dukascopy ticks inside missing minute intervals are added. "
            "No interpolation or forward filling."
        ),
        "missing_intervals_checked": len(intervals),
        "core_weekday_only": args.core_weekday_only,
        **download_report,
        **validate(args.output),
        "processing_seconds": time.time() - started,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
