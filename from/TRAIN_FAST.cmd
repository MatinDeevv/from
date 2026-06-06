@echo off
REM Quick training with optimized settings

build-vs\Release\from.exe train ^
  --data XAUUSD_ticks_all.parquet ^
  --config config_fast.toml ^
  --max-steps 100000 ^
  --no-ui
