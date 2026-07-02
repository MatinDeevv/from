#!/usr/bin/env python3
"""Convert modulo-timestamp XAUUSD quote ticks into execution-aware minute bars."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


UINT32_MS = 2**32
DEFAULT_BAR_MS = 60_000

PRICE_COLUMNS = ("bid", "ask", "mid")
OPEN_COLUMNS = tuple(f"{name}_open" for name in PRICE_COLUMNS) + ("spread_open",)
HIGH_COLUMNS = tuple(f"{name}_high" for name in PRICE_COLUMNS) + ("spread_high",)
LOW_COLUMNS = tuple(f"{name}_low" for name in PRICE_COLUMNS) + ("spread_low",)
CLOSE_COLUMNS = tuple(f"{name}_close" for name in PRICE_COLUMNS) + ("spread_close",)
MEAN_COLUMNS = ("spread_mean", "bid_vol_mean", "ask_vol_mean")

SCHEMA = pa.schema(
    [
        ("bar_index", pa.int64()),
        ("elapsed_start_ms", pa.int64()),
        ("first_tick_elapsed_ms", pa.int64()),
        ("last_tick_elapsed_ms", pa.int64()),
        ("gap_before_ms", pa.int64()),
        ("max_tick_gap_ms", pa.int64()),
        ("tick_count", pa.int32()),
        ("source_boundary_count", pa.int16()),
        ("raw_clock_wrap_count", pa.int16()),
        ("bid_open", pa.float64()),
        ("bid_high", pa.float64()),
        ("bid_low", pa.float64()),
        ("bid_close", pa.float64()),
        ("ask_open", pa.float64()),
        ("ask_high", pa.float64()),
        ("ask_low", pa.float64()),
        ("ask_close", pa.float64()),
        ("mid_open", pa.float64()),
        ("mid_high", pa.float64()),
        ("mid_low", pa.float64()),
        ("mid_close", pa.float64()),
        ("spread_open", pa.float64()),
        ("spread_high", pa.float64()),
        ("spread_low", pa.float64()),
        ("spread_close", pa.float64()),
        ("spread_mean", pa.float64()),
        ("bid_vol_mean", pa.float64()),
        ("ask_vol_mean", pa.float64()),
    ]
)


class BufferedWriter:
    def __init__(self, path: Path, flush_rows: int = 200_000) -> None:
        self.path = path
        self.flush_rows = flush_rows
        self.parts: dict[str, list[np.ndarray]] = {name: [] for name in SCHEMA.names}
        self.buffered_rows = 0
        self.total_rows = 0
        self.writer = pq.ParquetWriter(
            path,
            SCHEMA,
            compression="zstd",
            compression_level=6,
            use_dictionary=False,
            write_statistics=True,
        )

    def append(self, batch: dict[str, np.ndarray]) -> None:
        count = len(batch["bar_index"])
        if count == 0:
            return
        for name in SCHEMA.names:
            self.parts[name].append(batch[name])
        self.buffered_rows += count
        if self.buffered_rows >= self.flush_rows:
            self.flush()

    def append_record(self, record: dict[str, object]) -> None:
        batch = {name: np.asarray([record[name]]) for name in SCHEMA.names}
        self.append(batch)

    def flush(self) -> None:
        if self.buffered_rows == 0:
            return
        values = {name: np.concatenate(self.parts[name]) for name in SCHEMA.names}
        table = pa.Table.from_pydict(values, schema=SCHEMA)
        self.writer.write_table(table, row_group_size=self.flush_rows)
        self.total_rows += self.buffered_rows
        self.parts = {name: [] for name in SCHEMA.names}
        self.buffered_rows = 0

    def close(self) -> None:
        self.flush()
        self.writer.close()


def aggregate_bars(
    bar_index: np.ndarray,
    elapsed_ms: np.ndarray,
    tick_gap_ms: np.ndarray,
    boundary_flags: np.ndarray,
    wrap_flags: np.ndarray,
    bid: np.ndarray,
    ask: np.ndarray,
    mid: np.ndarray,
    bid_vol: np.ndarray,
    ask_vol: np.ndarray,
    bar_ms: int,
) -> dict[str, np.ndarray]:
    starts = np.r_[0, np.flatnonzero(bar_index[1:] != bar_index[:-1]) + 1]
    ends = np.r_[starts[1:], len(bar_index)]
    counts = ends - starts
    spread = ask - bid

    result: dict[str, np.ndarray] = {
        "bar_index": bar_index[starts].astype(np.int64),
        "elapsed_start_ms": (bar_index[starts] * bar_ms).astype(np.int64),
        "first_tick_elapsed_ms": elapsed_ms[starts].astype(np.int64),
        "last_tick_elapsed_ms": elapsed_ms[ends - 1].astype(np.int64),
        "gap_before_ms": tick_gap_ms[starts].astype(np.int64),
        "max_tick_gap_ms": np.maximum.reduceat(tick_gap_ms, starts).astype(np.int64),
        "tick_count": counts.astype(np.int32),
        "source_boundary_count": np.add.reduceat(boundary_flags, starts).astype(np.int16),
        "raw_clock_wrap_count": np.add.reduceat(wrap_flags, starts).astype(np.int16),
    }

    for name, values in (("bid", bid), ("ask", ask), ("mid", mid), ("spread", spread)):
        result[f"{name}_open"] = values[starts].astype(np.float64)
        result[f"{name}_high"] = np.maximum.reduceat(values, starts).astype(np.float64)
        result[f"{name}_low"] = np.minimum.reduceat(values, starts).astype(np.float64)
        result[f"{name}_close"] = values[ends - 1].astype(np.float64)

    result["spread_mean"] = (np.add.reduceat(spread, starts) / counts).astype(np.float64)
    result["bid_vol_mean"] = (np.add.reduceat(bid_vol, starts) / counts).astype(np.float64)
    result["ask_vol_mean"] = (np.add.reduceat(ask_vol, starts) / counts).astype(np.float64)
    return result


def take_record(batch: dict[str, np.ndarray], index: int) -> dict[str, object]:
    return {name: batch[name][index].item() for name in SCHEMA.names}


def slice_batch(batch: dict[str, np.ndarray], start: int, stop: int) -> dict[str, np.ndarray]:
    return {name: batch[name][start:stop] for name in SCHEMA.names}


def merge_pending_into_first(pending: dict[str, object], batch: dict[str, np.ndarray]) -> None:
    old_count = int(pending["tick_count"])
    new_count = int(batch["tick_count"][0])
    total_count = old_count + new_count

    for name in OPEN_COLUMNS:
        batch[name][0] = pending[name]
    for name in HIGH_COLUMNS:
        batch[name][0] = max(float(pending[name]), float(batch[name][0]))
    for name in LOW_COLUMNS:
        batch[name][0] = min(float(pending[name]), float(batch[name][0]))
    for name in MEAN_COLUMNS:
        batch[name][0] = (
            float(pending[name]) * old_count + float(batch[name][0]) * new_count
        ) / total_count

    batch["first_tick_elapsed_ms"][0] = pending["first_tick_elapsed_ms"]
    batch["gap_before_ms"][0] = pending["gap_before_ms"]
    batch["max_tick_gap_ms"][0] = max(
        int(pending["max_tick_gap_ms"]), int(batch["max_tick_gap_ms"][0])
    )
    batch["tick_count"][0] = total_count
    batch["source_boundary_count"][0] += int(pending["source_boundary_count"])
    batch["raw_clock_wrap_count"][0] += int(pending["raw_clock_wrap_count"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("cleaned"))
    parser.add_argument("--bar-ms", type=int, default=DEFAULT_BAR_MS)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bar_ms <= 0:
        raise ValueError("--bar-ms must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    bars_path = args.output_dir / "XAUUSD_1m_trade_bars.parquet"
    report_path = args.output_dir / "quality_report.json"

    for path in (bars_path, report_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    if args.overwrite and bars_path.exists():
        os.remove(bars_path)

    parquet = pq.ParquetFile(args.input)
    writer = BufferedWriter(bars_path)
    started = time.time()

    previous_raw_ms: int | None = None
    previous_valid_elapsed_ms: int | None = None
    origin_absolute_ms: int | None = None
    wrap_offset_ms = 0
    pending: dict[str, object] | None = None

    input_rows = valid_rows = invalid_rows = 0
    raw_clock_wraps = 0
    raw_clock_backwards_inside_group = 0
    source_boundaries = 0
    gap_counts = {">1s": 0, ">1m": 0, ">1h": 0, ">1d": 0, ">7d": 0}
    reason_counts = {
        "nonfinite": 0,
        "nonpositive_price": 0,
        "crossed_quote": 0,
        "negative_volume": 0,
        "mid_mismatch": 0,
    }
    min_price = math.inf
    max_price = -math.inf
    min_spread = math.inf
    max_spread = -math.inf
    spread_sum = 0.0
    max_tick_gap_ms = 0
    last_elapsed_ms = 0

    for group_id in range(parquet.num_row_groups):
        table = parquet.read_row_group(
            group_id, columns=["time", "ask", "bid", "mid", "ask_vol", "bid_vol"]
        )
        raw_ms = (
            table["time"].combine_chunks().cast(pa.int64()).to_numpy(zero_copy_only=False)
        )
        ask = table["ask"].combine_chunks().to_numpy(zero_copy_only=False)
        bid = table["bid"].combine_chunks().to_numpy(zero_copy_only=False)
        mid = table["mid"].combine_chunks().to_numpy(zero_copy_only=False)
        ask_vol = table["ask_vol"].combine_chunks().to_numpy(zero_copy_only=False)
        bid_vol = table["bid_vol"].combine_chunks().to_numpy(zero_copy_only=False)
        input_rows += len(raw_ms)

        inside_backwards = np.flatnonzero(np.diff(raw_ms) < 0)
        raw_clock_backwards_inside_group += len(inside_backwards)
        if len(inside_backwards):
            raise ValueError(
                f"Unexpected backward clock movement inside row group {group_id}; "
                "the current unwrapping model is not valid"
            )

        wrapped_at_boundary = previous_raw_ms is not None and raw_ms[0] < previous_raw_ms
        if wrapped_at_boundary:
            wrap_offset_ms += UINT32_MS
            raw_clock_wraps += 1
        if group_id > 0:
            source_boundaries += 1

        absolute_ms = raw_ms.astype(np.int64) + wrap_offset_ms
        if origin_absolute_ms is None:
            origin_absolute_ms = (int(absolute_ms[0]) // args.bar_ms) * args.bar_ms
        elapsed_ms = absolute_ms - origin_absolute_ms
        previous_raw_ms = int(raw_ms[-1])

        finite = (
            np.isfinite(ask)
            & np.isfinite(bid)
            & np.isfinite(mid)
            & np.isfinite(ask_vol)
            & np.isfinite(bid_vol)
        )
        positive_price = (ask > 0) & (bid > 0) & (mid > 0)
        uncrossed = ask >= bid
        nonnegative_volume = (ask_vol >= 0) & (bid_vol >= 0)
        midpoint_matches = np.abs(mid - (ask + bid) / 2.0) <= 1e-8
        valid = finite & positive_price & uncrossed & nonnegative_volume & midpoint_matches

        reason_counts["nonfinite"] += int(np.count_nonzero(~finite))
        reason_counts["nonpositive_price"] += int(np.count_nonzero(~positive_price))
        reason_counts["crossed_quote"] += int(np.count_nonzero(~uncrossed))
        reason_counts["negative_volume"] += int(np.count_nonzero(~nonnegative_volume))
        reason_counts["mid_mismatch"] += int(np.count_nonzero(~midpoint_matches))
        invalid_rows += int(np.count_nonzero(~valid))

        elapsed_valid = elapsed_ms[valid]
        ask_valid = ask[valid]
        bid_valid = bid[valid]
        mid_valid = mid[valid]
        ask_vol_valid = ask_vol[valid]
        bid_vol_valid = bid_vol[valid]
        valid_rows += len(elapsed_valid)
        if len(elapsed_valid) == 0:
            continue

        prepend = (
            int(elapsed_valid[0])
            if previous_valid_elapsed_ms is None
            else previous_valid_elapsed_ms
        )
        tick_gap_ms = np.diff(elapsed_valid, prepend=prepend).astype(np.int64)
        if np.any(tick_gap_ms < 0):
            raise ValueError(f"Reconstructed clock is not monotonic at row group {group_id}")
        previous_valid_elapsed_ms = int(elapsed_valid[-1])
        last_elapsed_ms = previous_valid_elapsed_ms

        gap_counts[">1s"] += int(np.count_nonzero(tick_gap_ms > 1_000))
        gap_counts[">1m"] += int(np.count_nonzero(tick_gap_ms > 60_000))
        gap_counts[">1h"] += int(np.count_nonzero(tick_gap_ms > 3_600_000))
        gap_counts[">1d"] += int(np.count_nonzero(tick_gap_ms > 86_400_000))
        gap_counts[">7d"] += int(np.count_nonzero(tick_gap_ms > 604_800_000))
        max_tick_gap_ms = max(max_tick_gap_ms, int(tick_gap_ms.max()))

        spread = ask_valid - bid_valid
        min_price = min(min_price, float(bid_valid.min()), float(ask_valid.min()))
        max_price = max(max_price, float(bid_valid.max()), float(ask_valid.max()))
        min_spread = min(min_spread, float(spread.min()))
        max_spread = max(max_spread, float(spread.max()))
        spread_sum += float(spread.sum())

        boundary_flags = np.zeros(len(elapsed_valid), dtype=np.int16)
        wrap_flags = np.zeros(len(elapsed_valid), dtype=np.int16)
        if group_id > 0:
            boundary_flags[0] = 1
        if wrapped_at_boundary:
            wrap_flags[0] = 1

        bars = aggregate_bars(
            elapsed_valid // args.bar_ms,
            elapsed_valid,
            tick_gap_ms,
            boundary_flags,
            wrap_flags,
            bid_valid,
            ask_valid,
            mid_valid,
            bid_vol_valid,
            ask_vol_valid,
            args.bar_ms,
        )

        if pending is not None:
            if int(bars["bar_index"][0]) == int(pending["bar_index"]):
                merge_pending_into_first(pending, bars)
            else:
                writer.append_record(pending)
            pending = None

        if len(bars["bar_index"]) > 1:
            writer.append(slice_batch(bars, 0, -1))
        pending = take_record(bars, -1)

        if group_id % 50 == 0:
            elapsed = time.time() - started
            print(
                f"row_group={group_id}/{parquet.num_row_groups} "
                f"rows={input_rows:,} bars_written={writer.total_rows:,} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    if pending is not None:
        writer.append_record(pending)
    writer.close()

    report = {
        "input": str(args.input.resolve()),
        "output": str(bars_path.resolve()),
        "bar_duration_ms": args.bar_ms,
        "input_rows": input_rows,
        "valid_rows": valid_rows,
        "invalid_rows_removed": invalid_rows,
        "invalid_reason_counts": reason_counts,
        "output_bars": writer.total_rows,
        "source_row_groups": parquet.num_row_groups,
        "source_boundaries": source_boundaries,
        "raw_clock_wraps_inferred": raw_clock_wraps,
        "raw_clock_backwards_inside_group": raw_clock_backwards_inside_group,
        "relative_elapsed_ms": last_elapsed_ms,
        "relative_elapsed_days": last_elapsed_ms / 86_400_000,
        "max_tick_gap_ms": max_tick_gap_ms,
        "gap_counts": gap_counts,
        "price_min": min_price,
        "price_max": max_price,
        "spread_min": min_spread,
        "spread_max": max_spread,
        "spread_mean": spread_sum / valid_rows,
        "clock_assumption": (
            "Raw timestamps are an unsigned 32-bit millisecond clock. Each backward "
            "row-group boundary is treated as one 2^32 ms rollover. Gaps longer than "
            "2^32 ms cannot be detected."
        ),
        "calendar_time_status": (
            "No calendar anchor exists in the source. Output time is truthful relative "
            "elapsed time only; no calendar dates were invented."
        ),
        "missing_bar_policy": "Minutes without ticks are omitted, not forward-filled.",
        "processing_seconds": time.time() - started,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
