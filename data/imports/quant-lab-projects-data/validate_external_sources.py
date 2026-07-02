#!/usr/bin/env python3
"""Validate restored XAUUSD calendar dates against independent external sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("bars", type=Path)
    parser.add_argument("--mapping", type=Path, default=Path("dated/row_group_calendar_mapping.csv"))
    parser.add_argument(
        "--world-bank",
        type=Path,
        default=Path("external_reference/world_bank_monthly.xlsx"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dated/multi_source_validation_report.json"),
    )
    return parser.parse_args()


def world_bank_gold(path: Path) -> pd.DataFrame:
    data = pd.read_excel(path, sheet_name="Monthly Prices", header=4)
    result = data.iloc[:, [0, data.columns.get_loc("Gold")]].copy()
    result.columns = ["month_code", "world_bank_gold"]
    result = result[result.month_code.astype(str).str.match(r"^\d{4}M\d{2}$")]
    result["month"] = pd.to_datetime(
        result.month_code.astype(str).str.replace("M", "-", regex=False) + "-01",
        utc=True,
    )
    result["world_bank_gold"] = pd.to_numeric(result.world_bank_gold, errors="coerce")
    return result.dropna(subset=["world_bank_gold"]).set_index("month")


def monthly_bars(path: Path, confident_only: bool) -> pd.DataFrame:
    connection = duckdb.connect()
    source = str(path).replace("\\", "/").replace("'", "''")
    where = "WHERE NOT timestamp_mapping_low_confidence" if confident_only else ""
    result = connection.execute(
        f"""
        WITH daily AS (
          SELECT
            cast(timestamp_utc AT TIME ZONE 'UTC' AS DATE) AS date,
            arg_max(mid_close, last_tick_time_utc) AS daily_close
          FROM read_parquet('{source}')
          {where}
          GROUP BY date
        )
        SELECT
          date_trunc('month', date)::DATE AS month,
          avg(daily_close) AS source_monthly_average,
          count(*) AS observed_days
        FROM daily
        GROUP BY month
        ORDER BY month
        """
    ).fetchdf()
    connection.close()
    result["month"] = pd.to_datetime(result.month, utc=True)
    return result.set_index("month")


def comparison_metrics(source: pd.DataFrame, reference: pd.DataFrame) -> dict:
    joined = source.join(reference[["world_bank_gold"]], how="inner").dropna()
    joined["level_log_error"] = np.abs(
        np.log(joined.source_monthly_average / joined.world_bank_gold)
    )
    joined["source_return"] = np.log(joined.source_monthly_average).diff()
    joined["reference_return"] = np.log(joined.world_bank_gold).diff()
    return {
        "matched_months": len(joined),
        "matched_start": joined.index.min().isoformat(),
        "matched_end": joined.index.max().isoformat(),
        "median_absolute_level_error_fraction": float(
            np.expm1(joined.level_log_error.median())
        ),
        "monthly_level_correlation": float(
            joined[["source_monthly_average", "world_bank_gold"]].corr().iloc[0, 1]
        ),
        "monthly_return_correlation": float(
            joined[["source_return", "reference_return"]].corr().iloc[0, 1]
        ),
    }


def main() -> None:
    args = parse_args()
    mapping = pd.read_csv(args.mapping)
    world_bank = world_bank_gold(args.world_bank)
    all_months = monthly_bars(args.bars, confident_only=False)
    confident_months = monthly_bars(args.bars, confident_only=True)

    corrections = mapping.loc[
        mapping.timestamp_dukascopy_cycle_corrected,
        ["source_row_group", "timestamp_cycle", "start_utc", "end_utc"],
    ].to_dict(orient="records")
    report = {
        "dukascopy": {
            "source_url_template": (
                "https://datafeed.dukascopy.com/datafeed/XAUUSD/"
                "{year}/{zero_based_month}/{day}/{hour}h_ticks.bi5"
            ),
            "source_row_groups": len(mapping),
            "exact_verified_source_row_groups": int(
                mapping.timestamp_dukascopy_exact_verified.sum()
            ),
            "corrected_source_row_groups": int(
                mapping.timestamp_dukascopy_cycle_corrected.sum()
            ),
            "corrections": corrections,
            "verification_method": (
                "Exact binary tick fingerprints matched against Dukascopy XAUUSD "
                "hourly .bi5 archive files."
            ),
        },
        "world_bank": {
            "reference": "World Bank Commodity Price Data (Pink Sheet), monthly Gold",
            "source_url": (
                "https://thedocs.worldbank.org/en/doc/"
                "18675f1d1639c7a34d463f59263ba0a2-0050012025/related/"
                "CMO-Historical-Data-Monthly.xlsx"
            ),
            "reference_file": str(args.world_bank.resolve()),
            "reference_start": world_bank.index.min().isoformat(),
            "reference_end": world_bank.index.max().isoformat(),
            "all_restored_dates": comparison_metrics(all_months, world_bank),
            "high_confidence_restored_dates": comparison_metrics(
                confident_months, world_bank
            ),
        },
        "interpretation": (
            "Dukascopy exact matches confirm or correct individual row-group cycles. "
            "World Bank monthly prices validate the broad calendar path but cannot "
            "certify individual tick timestamps."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
