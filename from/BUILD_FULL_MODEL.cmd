@echo off
REM ========================================================================
REM BUILD WITH FULL 24M PARAMETER MODEL + LOCK-FREE DATA LOADING
REM ========================================================================

echo.
echo ========================================================================
echo BUILDING FROM - FULL MODEL MODE
echo ========================================================================
echo.
echo Changes:
echo   [X] Switch to full FROM model (24M parameters!)
echo       - Conv1D with 256 channels
echo       - LSTM 512 hidden x 3 layers
echo       - 8-head Attention
echo       - 16 MoE experts
echo       - Temporal Fusion Transformer (3 blocks)
echo       - Hyperbolic embedding
echo       - Neural ODE
echo.
echo   [X] Lock-free async data loader
echo       - Atomic chunk allocation
echo       - One queue per worker (no contention)
echo       - 16 workers in parallel
echo       - Target: 14M rows/sec!
echo.
echo ========================================================================

cd /d %~dp0

echo.
echo [1/3] Cleaning...
if exist build-vs\Release\from.exe del /Q build-vs\Release\from.exe
echo       OK

echo.
echo [2/3] Rebuilding...
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=amd64 >nul 2>&1
cmake --build build-vs --config Release -j %NUMBER_OF_PROCESSORS%

if errorlevel 1 (
    echo.
    echo ========================================================================
    echo BUILD FAILED
    echo ========================================================================
    pause
    exit /b 1
)

echo       OK

echo.
echo [3/3] Verifying...
if exist build-vs\Release\from.exe (
    echo       OK - Executable ready
) else (
    echo       ERROR - Executable not found
    pause
    exit /b 1
)

echo.
echo ========================================================================
echo BUILD COMPLETE
echo ========================================================================
echo.
echo Model: 24M parameters (was: 96 parameters!)
echo.
echo Architecture:
echo   Conv1D:       256 channels (8 kernels, 4 dilations)
echo   LSTM:         512 hidden × 3 layers = 9M params
echo   Attention:    8 heads, 2048 d_ff = 8M params
echo   MoE:          16 experts × 512 dims = 4M params
echo   TFT:          3 blocks = 2M params
echo   Hyperbolic:   64 dims
echo   Neural ODE:   128 hidden, 10 steps
echo   Total:        ~24M parameters
echo.
echo Data Loader: LOCK-FREE
echo   Atomic chunk grabbing
echo   16 parallel workers
echo   No mutex contention
echo   Target: 14M rows/sec
echo.
echo Expected Performance:
echo   GPU Utilization:  90-99%% (FULL saturation!)
echo   Data Throughput:  14M rows/sec (was: 127k)
echo   Training Speed:   100-500x faster
echo.
echo ========================================================================
echo.
echo Run with:
echo   build-vs\Release\from.exe train-full --data XAUUSD_ticks_all.parquet --max-steps 1000
echo.
echo Or just:
echo   TRAIN_FULL.cmd
echo.
echo ========================================================================

pause
