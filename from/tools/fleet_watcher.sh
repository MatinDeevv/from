#!/usr/bin/env bash
# Waits for the 4-year tick fetch to finish, then auto-launches the FULL committee
# fleet (mlp + deep, 324 configs) on the dense dataset into a clean ~/registry.
# Run detached:  nohup bash fleet_watcher.sh > ~/fleet_watcher.log 2>&1 &
set -uo pipefail
DATA_HOST="$HOME/data/XAUUSD_ticks_all.parquet"
DATA_CTR="/data/XAUUSD_ticks_all.parquet"

echo "[watcher] $(date -u +%FT%TZ) waiting for fetch (pgrep ticks_all) to finish ..."
while pgrep -f ticks_all >/dev/null 2>&1; do sleep 60; done
echo "[watcher] $(date -u +%FT%TZ) fetch process gone."

# Give the final parquet write a moment, then verify it exists and is non-trivial.
sleep 20
if [ ! -f "$DATA_HOST" ]; then
  echo "[watcher] FATAL: $DATA_HOST not found — fetch likely died before writing. Aborting."
  exit 1
fi
sz=$(stat -c%s "$DATA_HOST"); echo "[watcher] dense dataset ready: $sz bytes"
if [ "$sz" -lt 100000000 ]; then
  echo "[watcher] WARN: dataset < 100MB — fetch may be incomplete, but proceeding."
fi

# Clean committee registry, then run the full fleet on dense data.
rm -rf "$HOME/registry"; mkdir -p "$HOME/registry"
echo "[watcher] $(date -u +%FT%TZ) launching FULL fleet (324 configs) on dense data ..."
# max-samples 2,500,000 keeps the deep model's resident tick stream (~160M ticks) under
# the L4 24GB cap; steps 8000 with epoch budget 14.
REGISTRY="$HOME/registry" bash "$HOME/sweep_fleet.sh" "$DATA_CTR" 2500000 8000
echo "[watcher] $(date -u +%FT%TZ) FULL fleet finished. Registry: $HOME/registry"
