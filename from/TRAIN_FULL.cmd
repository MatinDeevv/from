@echo off
REM Train with full 24M parameter model

echo.
echo ========================================================================
echo TRAINING FULL FROM MODEL - 24M PARAMETERS
echo ========================================================================
echo.

if not exist build-vs\Release\from.exe (
    echo Build not found! Run BUILD_FULL_MODEL.cmd first
    pause
    exit /b 1
)

build-vs\Release\from.exe train-full ^
  --data XAUUSD_ticks_all.parquet ^
  --max-steps 10000 ^
  --no-ui

pause
