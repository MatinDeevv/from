#!/usr/bin/env python3
"""Restore modulo-2^32 XAUUSD timestamps using external gold references."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yfinance as yf


UINT32_MS = 2**32
DAY_MS = 86_400_000
MINUTE_MS = 60_000
LOW_CONFIDENCE_SCORE = 0.01
LOW_CONFIDENCE_MARGIN = 0.001

PRICE_NAMES = ("bid", "ask", "mid", "spread")
PARTIAL_SCHEMA = pa.schema(
    [
        ("bar_start_epoch_ms", pa.int64()),
        ("first_tick_epoch_ms", pa.int64()),
        ("last_tick_epoch_ms", pa.int64()),
        ("max_source_tick_gap_ms", pa.int64()),
        ("source_row_group", pa.int32()),
        ("timestamp_cycle", pa.int16()),
        ("timestamp_mapping_score", pa.float32()),
        ("timestamp_mapping_low_confidence", pa.bool_()),
        ("timestamp_dukascopy_exact_verified", pa.bool_()),
        ("timestamp_dukascopy_cycle_corrected", pa.bool_()),
        ("tick_count", pa.int32()),
        *[
            (f"{name}_{field}", pa.float64())
            for name in PRICE_NAMES
            for field in ("open", "high", "low", "close")
        ],
        ("spread_mean", pa.float64()),
        ("bid_vol_mean", pa.float64()),
        ("ask_vol_mean", pa.float64()),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("dated"))
    parser.add_argument("--ticker", default="GC=F")
    parser.add_argument("--dukascopy-verification", type=Path)
    parser.add_argument("--dukascopy-multi-verification", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def yahoo_reference(ticker: str, output_dir: Path) -> pd.DataFrame:
    data = yf.download(
        ticker,
        period="max",
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if data.empty:
        raise RuntimeError(f"Yahoo returned no data for {ticker}")
    if isinstance(data.columns, pd.MultiIndex):
        data = data.xs(ticker, axis=1, level="Ticker")
    data.index = pd.to_datetime(data.index, utc=True)
    data = data[["Open", "High", "Low", "Close"]].dropna()
    data.to_csv(output_dir / "yahoo_GC_F_daily.csv")
    return data


def score_row_groups(
    parquet: pq.ParquetFile,
    reference: pd.DataFrame,
    latest_allowed_ms: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    min_day = int(reference.index.min().timestamp() * 1000 // DAY_MS)
    max_day = int(reference.index.max().timestamp() * 1000 // DAY_MS)
    lows = np.full(max_day - min_day + 1, np.nan)
    highs = np.full_like(lows, np.nan)
    closes = np.full_like(lows, np.nan)
    for date, row in reference.iterrows():
        index = int(date.timestamp() * 1000 // DAY_MS) - min_day
        lows[index], highs[index], closes[index] = row["Low"], row["High"], row["Close"]

    first_reference_ms = int(reference.index.min().timestamp() * 1000)
    min_cycle = max(0, first_reference_ms // UINT32_MS - 2)
    max_cycle = latest_allowed_ms // UINT32_MS + 1
    cycles = np.arange(min_cycle, max_cycle + 1, dtype=np.int64)
    scores = np.full((parquet.num_row_groups, len(cycles)), 1.0)
    inside = np.zeros_like(scores)
    matched = np.zeros_like(scores, dtype=np.int16)
    starts = np.zeros(parquet.num_row_groups, dtype=np.int64)
    ends = np.zeros(parquet.num_row_groups, dtype=np.int64)

    for group_id in range(parquet.num_row_groups):
        table = parquet.read_row_group(group_id, columns=["time", "mid"])
        count = table.num_rows
        sample_index = np.unique(np.linspace(0, count - 1, min(301, count), dtype=int))
        raw_ms = (
            table["time"].combine_chunks().cast(pa.int64()).to_numpy(zero_copy_only=False)
        )
        mids = table["mid"].combine_chunks().to_numpy(zero_copy_only=False)
        starts[group_id], ends[group_id] = raw_ms[0], raw_ms[-1]
        raw_ms, mids = raw_ms[sample_index], mids[sample_index]

        for cycle_index, cycle in enumerate(cycles):
            candidate_end = int(ends[group_id] + cycle * UINT32_MS)
            if candidate_end > latest_allowed_ms:
                continue
            day_index = (raw_ms + cycle * UINT32_MS) // DAY_MS - min_day
            valid = (day_index >= 0) & (day_index < len(lows))
            available = np.zeros(len(day_index), dtype=bool)
            available[valid] = np.isfinite(lows[day_index[valid]])
            found = int(available.sum())
            matched[group_id, cycle_index] = found
            if found < max(10, len(day_index) // 3):
                continue

            values = mids[available]
            low = lows[day_index[available]]
            high = highs[day_index[available]]
            close = closes[day_index[available]]
            range_error = np.where(
                values < low,
                np.log(low / values),
                np.where(values > high, np.log(values / high), 0.0),
            )
            close_error = np.abs(np.log(values / close))
            inside_fraction = np.mean((values >= low) & (values <= high))
            inside[group_id, cycle_index] = inside_fraction
            scores[group_id, cycle_index] = (
                np.mean(range_error)
                + 0.08 * np.mean(close_error)
                + 0.002 * (1 - found / len(day_index))
            )

    # Row groups overlap and some appear out of order, so align independently.
    path = np.argmin(scores, axis=1)
    chosen_cycles = cycles[path]

    rows = []
    for group_id, state in enumerate(path):
        order = np.argsort(scores[group_id])
        second_state = int(order[1])
        ambiguity_margin = float(scores[group_id, second_state] - scores[group_id, state])
        low_confidence = bool(
            scores[group_id, state] > LOW_CONFIDENCE_SCORE
            or ambiguity_margin < LOW_CONFIDENCE_MARGIN
        )
        rows.append(
            {
                "source_row_group": group_id,
                "timestamp_cycle": int(chosen_cycles[group_id]),
                "mapping_score": float(scores[group_id, state]),
                "mapping_low_confidence": low_confidence,
                "mapping_ambiguity_margin": ambiguity_margin,
                "inside_yahoo_daily_range_fraction": float(inside[group_id, state]),
                "matched_samples": int(matched[group_id, state]),
                "start_utc": pd.to_datetime(
                    starts[group_id] + chosen_cycles[group_id] * UINT32_MS,
                    unit="ms",
                    utc=True,
                ),
                "end_utc": pd.to_datetime(
                    ends[group_id] + chosen_cycles[group_id] * UINT32_MS,
                    unit="ms",
                    utc=True,
                ),
                "local_best_cycle": int(cycles[order[0]]),
                "local_best_score": float(scores[group_id, order[0]]),
                "second_best_cycle": int(cycles[second_state]),
                "second_best_score": float(scores[group_id, second_state]),
            }
        )
    return pd.DataFrame(rows), chosen_cycles


def apply_dukascopy_verification(
    mapping: pd.DataFrame,
    first_path: Path | None,
    multi_path: Path | None,
) -> pd.DataFrame:
    mapping = mapping.copy()
    mapping["timestamp_dukascopy_exact_verified"] = False
    mapping["timestamp_dukascopy_cycle_corrected"] = False
    mapping["dukascopy_exact_fingerprint_matches"] = 0
    old_cycles = mapping.timestamp_cycle.copy()

    if first_path is not None and first_path.exists():
        first = pd.read_csv(first_path).set_index("source_row_group")
        unique = first.verified_cycle.notna()
        unique_ids = first.index[unique]
        verified_cycles = first.loc[unique_ids, "verified_cycle"].astype(int)
        mapping.loc[unique_ids, "timestamp_cycle"] = verified_cycles
        mapping.loc[unique_ids, "timestamp_dukascopy_exact_verified"] = True
        mapping.loc[unique_ids, "dukascopy_exact_fingerprint_matches"] = first.loc[
            unique_ids, "exact_match_count"
        ].astype(int)

    if multi_path is not None and multi_path.exists():
        multi = pd.read_csv(multi_path).set_index("source_row_group")
        selected = multi[
            multi.dukascopy_multi_verified
            & (
                multi.inferred_cycle
                == mapping.loc[multi.index, "timestamp_cycle"].to_numpy()
            )
        ]
        mapping.loc[selected.index, "timestamp_dukascopy_exact_verified"] = True
        mapping.loc[selected.index, "dukascopy_exact_fingerprint_matches"] = np.maximum(
            mapping.loc[selected.index, "dukascopy_exact_fingerprint_matches"].to_numpy(),
            selected.exact_fingerprint_matches.to_numpy(),
        )

    corrected = mapping.timestamp_cycle != old_cycles
    mapping.loc[corrected, "timestamp_dukascopy_cycle_corrected"] = True
    mapping.loc[mapping.timestamp_dukascopy_exact_verified, "mapping_low_confidence"] = False

    cycle_delta = (mapping.timestamp_cycle - old_cycles) * UINT32_MS
    mapping["start_utc"] = pd.to_datetime(
        mapping.start_utc, utc=True, format="mixed"
    ) + pd.to_timedelta(cycle_delta, unit="ms")
    mapping["end_utc"] = pd.to_datetime(
        mapping.end_utc, utc=True, format="mixed"
    ) + pd.to_timedelta(cycle_delta, unit="ms")
    return mapping


def aggregate_partial(
    actual_ms: np.ndarray,
    bid: np.ndarray,
    ask: np.ndarray,
    mid: np.ndarray,
    bid_vol: np.ndarray,
    ask_vol: np.ndarray,
    group_id: int,
    cycle: int,
    score: float,
    low_confidence: bool,
    dukascopy_exact_verified: bool,
    dukascopy_cycle_corrected: bool,
) -> pa.Table:
    bar = actual_ms // MINUTE_MS
    starts = np.r_[0, np.flatnonzero(bar[1:] != bar[:-1]) + 1]
    ends = np.r_[starts[1:], len(bar)]
    counts = ends - starts
    gaps = np.diff(actual_ms, prepend=actual_ms[0])
    spread = ask - bid

    data: dict[str, np.ndarray] = {
        "bar_start_epoch_ms": (bar[starts] * MINUTE_MS).astype(np.int64),
        "first_tick_epoch_ms": actual_ms[starts].astype(np.int64),
        "last_tick_epoch_ms": actual_ms[ends - 1].astype(np.int64),
        "max_source_tick_gap_ms": np.maximum.reduceat(gaps, starts).astype(np.int64),
        "source_row_group": np.full(len(starts), group_id, dtype=np.int32),
        "timestamp_cycle": np.full(len(starts), cycle, dtype=np.int16),
        "timestamp_mapping_score": np.full(len(starts), score, dtype=np.float32),
        "timestamp_mapping_low_confidence": np.full(
            len(starts), low_confidence, dtype=bool
        ),
        "timestamp_dukascopy_exact_verified": np.full(
            len(starts), dukascopy_exact_verified, dtype=bool
        ),
        "timestamp_dukascopy_cycle_corrected": np.full(
            len(starts), dukascopy_cycle_corrected, dtype=bool
        ),
        "tick_count": counts.astype(np.int32),
    }
    for name, values in (("bid", bid), ("ask", ask), ("mid", mid), ("spread", spread)):
        data[f"{name}_open"] = values[starts].astype(np.float64)
        data[f"{name}_high"] = np.maximum.reduceat(values, starts).astype(np.float64)
        data[f"{name}_low"] = np.minimum.reduceat(values, starts).astype(np.float64)
        data[f"{name}_close"] = values[ends - 1].astype(np.float64)
    data["spread_mean"] = np.add.reduceat(spread, starts) / counts
    data["bid_vol_mean"] = np.add.reduceat(bid_vol, starts) / counts
    data["ask_vol_mean"] = np.add.reduceat(ask_vol, starts) / counts
    return pa.Table.from_pydict(data, schema=PARTIAL_SCHEMA)


def write_partial_bars(
    parquet: pq.ParquetFile,
    mapping: pd.DataFrame,
    cycles: np.ndarray,
    path: Path,
) -> None:
    writer = pq.ParquetWriter(path, PARTIAL_SCHEMA, compression="zstd", compression_level=6)
    for group_id, cycle in enumerate(cycles):
        table = parquet.read_row_group(
            group_id, columns=["time", "bid", "ask", "mid", "bid_vol", "ask_vol"]
        )
        raw_ms = (
            table["time"].combine_chunks().cast(pa.int64()).to_numpy(zero_copy_only=False)
        )
        arrays = [
            table[name].combine_chunks().to_numpy(zero_copy_only=False)
            for name in ("bid", "ask", "mid", "bid_vol", "ask_vol")
        ]
        partial = aggregate_partial(
            raw_ms + int(cycle) * UINT32_MS,
            *arrays,
            group_id,
            int(cycle),
            float(mapping.loc[group_id, "mapping_score"]),
            bool(mapping.loc[group_id, "mapping_low_confidence"]),
            bool(mapping.loc[group_id, "timestamp_dukascopy_exact_verified"]),
            bool(mapping.loc[group_id, "timestamp_dukascopy_cycle_corrected"]),
        )
        writer.write_table(partial, row_group_size=200_000)
        if group_id % 50 == 0:
            print(f"partial bars: row group {group_id}/{parquet.num_row_groups}", flush=True)
    writer.close()


def merge_partial_bars(partial_path: Path, final_path: Path) -> None:
    connection = duckdb.connect()
    connection.execute("SET preserve_insertion_order=false")
    source = str(partial_path).replace("\\", "/").replace("'", "''")
    destination = str(final_path).replace("\\", "/").replace("'", "''")
    query = f"""
    COPY (
      WITH merged AS (
        SELECT
          bar_start_epoch_ms,
          min(first_tick_epoch_ms) AS first_tick_epoch_ms,
          max(last_tick_epoch_ms) AS last_tick_epoch_ms,
          max(max_source_tick_gap_ms) AS max_source_tick_gap_ms,
          sum(tick_count)::BIGINT AS tick_count,
          count(*)::INTEGER AS source_group_count,
          max(timestamp_mapping_score)::FLOAT AS timestamp_mapping_score_max,
          bool_or(timestamp_mapping_low_confidence) AS timestamp_mapping_low_confidence,
          bool_and(timestamp_dukascopy_exact_verified)
            AS timestamp_dukascopy_exact_verified_all,
          bool_or(timestamp_dukascopy_exact_verified)
            AS timestamp_dukascopy_exact_verified_any,
          bool_or(timestamp_dukascopy_cycle_corrected)
            AS timestamp_dukascopy_cycle_corrected_any,
          arg_min(bid_open, first_tick_epoch_ms) AS bid_open,
          max(bid_high) AS bid_high,
          min(bid_low) AS bid_low,
          arg_max(bid_close, last_tick_epoch_ms) AS bid_close,
          arg_min(ask_open, first_tick_epoch_ms) AS ask_open,
          max(ask_high) AS ask_high,
          min(ask_low) AS ask_low,
          arg_max(ask_close, last_tick_epoch_ms) AS ask_close,
          arg_min(mid_open, first_tick_epoch_ms) AS mid_open,
          max(mid_high) AS mid_high,
          min(mid_low) AS mid_low,
          arg_max(mid_close, last_tick_epoch_ms) AS mid_close,
          arg_min(spread_open, first_tick_epoch_ms) AS spread_open,
          max(spread_high) AS spread_high,
          min(spread_low) AS spread_low,
          arg_max(spread_close, last_tick_epoch_ms) AS spread_close,
          sum(spread_mean * tick_count) / sum(tick_count) AS spread_mean,
          sum(bid_vol_mean * tick_count) / sum(tick_count) AS bid_vol_mean,
          sum(ask_vol_mean * tick_count) / sum(tick_count) AS ask_vol_mean
        FROM read_parquet('{source}')
        GROUP BY bar_start_epoch_ms
      ),
      gaps AS (
        SELECT *,
          first_tick_epoch_ms
            - lag(last_tick_epoch_ms) OVER (ORDER BY bar_start_epoch_ms) AS gap_before_ms,
          bar_start_epoch_ms
            - lag(bar_start_epoch_ms) OVER (ORDER BY bar_start_epoch_ms)
            - {MINUTE_MS} AS missing_time_before_ms
        FROM merged
      )
      SELECT
        timezone('UTC', epoch_ms(bar_start_epoch_ms)) AS timestamp_utc,
        timezone('UTC', epoch_ms(first_tick_epoch_ms)) AS first_tick_time_utc,
        timezone('UTC', epoch_ms(last_tick_epoch_ms)) AS last_tick_time_utc,
        coalesce(gap_before_ms, 0)::BIGINT AS gap_before_ms,
        greatest(coalesce(missing_time_before_ms, 0), 0)::BIGINT AS missing_time_before_ms,
        max_source_tick_gap_ms,
        tick_count,
        source_group_count,
        timestamp_mapping_score_max,
        timestamp_mapping_low_confidence,
        timestamp_dukascopy_exact_verified_all,
        timestamp_dukascopy_exact_verified_any,
        timestamp_dukascopy_cycle_corrected_any,
        bid_open, bid_high, bid_low, bid_close,
        ask_open, ask_high, ask_low, ask_close,
        mid_open, mid_high, mid_low, mid_close,
        spread_open, spread_high, spread_low, spread_close,
        spread_mean, bid_vol_mean, ask_vol_mean
      FROM gaps
      ORDER BY bar_start_epoch_ms
    ) TO '{destination}'
      (FORMAT PARQUET, COMPRESSION ZSTD, COMPRESSION_LEVEL 6, ROW_GROUP_SIZE 200000)
    """
    connection.execute(query)
    connection.close()


def validate(final_path: Path, reference: pd.DataFrame, mapping: pd.DataFrame) -> dict:
    connection = duckdb.connect()
    path = str(final_path).replace("\\", "/").replace("'", "''")
    totals = connection.execute(
        f"""
        SELECT count(*), sum(tick_count),
          min(timestamp_utc AT TIME ZONE 'UTC'),
          max(last_tick_time_utc AT TIME ZONE 'UTC'),
          count(*) FILTER (WHERE timestamp_mapping_low_confidence),
          count(*) FILTER (WHERE source_group_count > 1),
          count(*) FILTER (WHERE timestamp_dukascopy_exact_verified_all),
          count(*) FILTER (WHERE timestamp_dukascopy_exact_verified_any),
          count(*) FILTER (WHERE timestamp_dukascopy_cycle_corrected_any),
          count(*) FILTER (WHERE ask_open < bid_open OR ask_close < bid_close),
          count(*) FILTER (
            WHERE mid_low > least(mid_open, mid_close)
               OR mid_high < greatest(mid_open, mid_close)
          )
        FROM read_parquet('{path}')
        """
    ).fetchone()
    daily = connection.execute(
        f"""
        SELECT cast(timestamp_utc AT TIME ZONE 'UTC' AS DATE) AS date,
          arg_max(mid_close, last_tick_time_utc) AS source_close
        FROM read_parquet('{path}')
        GROUP BY date ORDER BY date
        """
    ).fetchdf()
    confident_daily = connection.execute(
        f"""
        SELECT cast(timestamp_utc AT TIME ZONE 'UTC' AS DATE) AS date,
          arg_max(mid_close, last_tick_time_utc) AS source_close
        FROM read_parquet('{path}')
        WHERE NOT timestamp_mapping_low_confidence
        GROUP BY date ORDER BY date
        """
    ).fetchdf()
    connection.close()

    daily["date"] = pd.to_datetime(daily["date"], utc=True)
    joined = daily.set_index("date").join(reference[["High", "Low", "Close"]], how="inner")
    joined["inside"] = (joined.source_close >= joined.Low) & (joined.source_close <= joined.High)
    joined["close_log_error"] = np.abs(np.log(joined.source_close / joined.Close))
    joined["source_return"] = np.log(joined.source_close).diff()
    joined["yahoo_return"] = np.log(joined.Close).diff()
    confident_daily["date"] = pd.to_datetime(confident_daily["date"], utc=True)
    confident_joined = confident_daily.set_index("date").join(
        reference[["High", "Low", "Close"]], how="inner"
    )
    confident_joined["source_return"] = np.log(confident_joined.source_close).diff()
    confident_joined["yahoo_return"] = np.log(confident_joined.Close).diff()

    return {
        "output_bars": totals[0],
        "ticks_reconciled": totals[1],
        "start_utc": f"{totals[2].isoformat()}Z",
        "end_utc": f"{totals[3].isoformat()}Z",
        "low_confidence_bars": totals[4],
        "bars_merged_from_multiple_source_groups": totals[5],
        "dukascopy_exact_verified_all_bars": totals[6],
        "dukascopy_exact_verified_any_bars": totals[7],
        "dukascopy_corrected_bars": totals[8],
        "crossed_open_or_close_bars": totals[9],
        "bad_mid_ohlc_bars": totals[10],
        "source_row_groups": len(mapping),
        "low_confidence_source_row_groups": int(mapping.mapping_low_confidence.sum()),
        "dukascopy_exact_verified_source_row_groups": int(
            mapping.timestamp_dukascopy_exact_verified.sum()
        ),
        "dukascopy_corrected_source_row_groups": int(
            mapping.timestamp_dukascopy_cycle_corrected.sum()
        ),
        "mapping_score_quantiles": {
            str(key): float(value)
            for key, value in mapping.mapping_score.quantile(
                [0, 0.5, 0.9, 0.95, 0.99, 1]
            ).items()
        },
        "yahoo_reference_ticker": "GC=F",
        "yahoo_reference_start": str(reference.index.min()),
        "yahoo_reference_end": str(reference.index.max()),
        "matched_daily_closes": len(joined),
        "daily_close_inside_yahoo_range_fraction": float(joined.inside.mean()),
        "median_daily_close_error_fraction": float(
            np.expm1(joined.close_log_error.median())
        ),
        "daily_return_correlation_with_yahoo": float(
            joined[["source_return", "yahoo_return"]].corr().iloc[0, 1]
        ),
        "high_confidence_matched_daily_closes": len(confident_joined),
        "high_confidence_daily_return_correlation_with_yahoo": float(
            confident_joined[["source_return", "yahoo_return"]].corr().iloc[0, 1]
        ),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_path = args.output_dir / "XAUUSD_1m_dated_trade_bars.parquet"
    mapping_path = args.output_dir / "row_group_calendar_mapping.csv"
    report_path = args.output_dir / "calendar_restoration_report.json"
    partial_path = args.output_dir / "_partial_bars.parquet"

    for path in (final_path, mapping_path, report_path, partial_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
        if path.exists():
            path.unlink()

    started = time.time()
    reference = yahoo_reference(args.ticker, args.output_dir)
    reference_end_ms = int((reference.index.max() + pd.Timedelta(days=1)).timestamp() * 1000)
    latest_allowed_ms = min(int(args.input.stat().st_mtime * 1000), reference_end_ms)
    parquet = pq.ParquetFile(args.input)
    mapping, cycles = score_row_groups(parquet, reference, latest_allowed_ms)
    first_verification = args.dukascopy_verification
    multi_verification = args.dukascopy_multi_verification
    if first_verification is None:
        candidate = args.output_dir / "dukascopy_verification.csv"
        first_verification = candidate if candidate.exists() else None
    if multi_verification is None:
        candidate = args.output_dir / "dukascopy_multi_verification.csv"
        multi_verification = candidate if candidate.exists() else None
    mapping = apply_dukascopy_verification(
        mapping, first_verification, multi_verification
    )
    cycles = mapping.timestamp_cycle.to_numpy(dtype=np.int64)
    mapping.to_csv(mapping_path, index=False)
    write_partial_bars(parquet, mapping, cycles, partial_path)
    merge_partial_bars(partial_path, final_path)
    partial_path.unlink()

    report = validate(final_path, reference, mapping)
    report.update(
        {
            "input": str(args.input.resolve()),
            "output": str(final_path.resolve()),
            "timestamp_model": (
                "Raw time equals real Unix epoch milliseconds modulo 2^32. "
                "Exact Dukascopy tick fingerprints override the independently "
                "best-matching Yahoo GC=F epoch cycle when available."
            ),
            "dukascopy_verification": (
                str(first_verification.resolve()) if first_verification else None
            ),
            "dukascopy_multi_verification": (
                str(multi_verification.resolve()) if multi_verification else None
            ),
            "low_confidence_score_threshold": LOW_CONFIDENCE_SCORE,
            "low_confidence_ambiguity_margin_threshold": LOW_CONFIDENCE_MARGIN,
            "latest_allowed_utc": str(
                pd.to_datetime(latest_allowed_ms, unit="ms", utc=True)
            ),
            "processing_seconds": time.time() - started,
        }
    )
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
