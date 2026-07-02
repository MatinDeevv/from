# XAUUSD trading data

Use:

```text
MASTER_DATASET.parquet
```

This is the single canonical trading dataset. It contains only the
high-confidence rows from the restored calendar, sorted by UTC timestamp, with
gap diagnostics recalculated after filtering.

The files under `cleaned/` and `dated/` are intermediate and audit outputs, not
the master trading dataset.

## Restored calendar

The raw timestamp is consistent with Unix epoch milliseconds truncated modulo
`2^32`. `restore_xauusd_calendar.py` compares each source row group against
Yahoo Finance daily gold futures (`GC=F`) and independently selects its
best-matching epoch cycle. Exact binary tick fingerprints from Dukascopy's
XAUUSD archive override those statistical assignments where available.
Overlapping row groups are then merged into globally sorted UTC one-minute
bars.

Run:

```powershell
py restore_xauusd_calendar.py XAUUSD_ticks_all.parquet --output-dir dated --overwrite
```

The master dataset span is:

```text
2003-05-07 00:26:00 UTC to 2026-05-22 13:54:31.113 UTC
```

The broader audit output includes an inferred low-confidence segment beginning:

```text
2003-01-27 14:17:00 UTC
```

## Confidence

`MASTER_DATASET.parquet` already excludes every row marked
`timestamp_mapping_low_confidence`.

For a strict exact-archive-confirmed set, require:

```sql
WHERE timestamp_dukascopy_exact_verified_all
```

The master dataset contains:

- `4,873,278` high-confidence one-minute bars.
- `468,421,246` retained source ticks.
- `264` exactly Dukascopy-verified source row groups out of `685`.
- `3` source row-group cycles corrected by exact Dukascopy matches.
- `0` low-confidence bars.
- `2,165,374` bars for which every contributing group is exactly verified.
- `0.204%` high-confidence monthly median difference versus World Bank Gold.
- `0.901` high-confidence monthly-return correlation versus World Bank Gold.

Yahoo gold futures and World Bank Gold are reference series, not the same
instrument as spot XAUUSD. Dukascopy exact matches certify the selected cycle
for those source groups only. The remaining groups are statistically inferred,
not exchange-certified.

Independent validation details are in:

```text
dated/multi_source_validation_report.json
```

## Columns

`timestamp_utc`, `first_tick_time_utc`, and `last_tick_time_utc` are timezone-
aware UTC timestamps. Some clients display them in the machine's local
timezone.

Each bar includes:

- Bid, ask, mid, and spread OHLC.
- Mean spread and mean displayed bid/ask size.
- Tick and contributing-source-group counts.
- Gap and missing-time diagnostics.
- `timestamp_mapping_score_max`.
- `timestamp_mapping_low_confidence`.
- `timestamp_dukascopy_exact_verified_all`.
- `timestamp_dukascopy_exact_verified_any`.
- `timestamp_dukascopy_cycle_corrected_any`.

Use ask prices for simulated buys and bid prices for simulated sells.
