# Final verdict: Medallion-Lite VM run

The 8xL4 VM expired after completing 24 of 324 dense-fleet configurations.
The queue was horizon-ascending, so every completed dense model used the
shortest, h1024 horizon. The longer h6144-h12288 configurations were not run.

## Evidence

- All 24 completed h1024 models were significant losers after cost: profit
  factor 0.10-0.22 and t-statistics from -1.4 to -12.7.
- The sparse 2023 fleet covered 162 models. None cleared its null bar.
- In the sparse run, long horizons reached point-estimate profit factors around
  1.1-1.46, but were at or below the null distribution: PF p95=1.09,
  p99=1.83; t p95=0.17, p99=0.41.

## Conclusion

No demonstrable edge was found. Short horizons are rejected. Dense
long-horizon performance remains untested, so it should not be represented as
a confirmed negative result.

## Next run

1. Pre-build each cache once.
2. Run long horizons first (h12288 down to h1024), with deep models before MLPs.
3. Evaluate against a sufficiently powered null distribution and FDR control.
4. Require independent forward paper testing for any survivor.
