@echo off
REM Ultra-fast training with all optimizations

echo.
echo ========================================================================
echo MAXIMUM PERFORMANCE TRAINING
echo ========================================================================
echo.

if not exist build-max\from.exe (
    echo Build not found! Run BUILD_MAXIMUM_PERFORMANCE.cmd first
    pause
    exit /b 1
)

build-max\from.exe train ^
  --data XAUUSD_ticks_all.parquet ^
  --batch-size 8192 ^
  --max-steps 100000 ^
  --no-ui

pause
