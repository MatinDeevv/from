# After-train VM snapshot

This directory records the useful, portable conclusions from the 2026-06-21
Medallion-Lite XAUUSD training fleet. The scripts that produced the fleet are
already maintained in `../tools/`, and the newer implementations of the
walk-forward runners live in `../cli/`.

The original local snapshot at `C:\Users\marti\Desktop\after_train` also
contains raw tick data, model weights, cache files, and VM output copies. They
are intentionally not versioned: the data is roughly 2.4 GB, weights are
derived artifacts, and the duplicate VM copies are not source-of-truth code.

Tracked documents:

- `FINAL_VERDICT.md`: scope, completed coverage, and evidence-bound conclusion.
- `executive_summary.md`: compact decision summary for the completed dense run.

For a future VM run, train long horizons first and pre-build caches once. A
candidate must clear the null-distribution threshold and multiple-testing
control before it can be considered for forward paper testing.
