@echo off
cd /d C:\Users\marti\from\from

taskkill /F /IM from.exe >NUL 2>&1

set "PATH="
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=amd64 -host_arch=amd64 >NUL 2>&1

REM Delete stale cache (config changed: horizon=256, threshold=2.0)
if exist XAUUSD_ticks_all.parquet.cache (
    echo Deleting old cache (config changed)...
    del /F XAUUSD_ticks_all.parquet.cache
)

REM Configure CUDA build if not done yet
if not exist build-cuda\CMakeCache.txt (
    echo Configuring CUDA build...
    cmake -B build-cuda -G "Visual Studio 17 2022" -A x64 -DFROM_CUDA=ON -DFROM_NATIVE_ARCH=ON -DCMAKE_BUILD_TYPE=Release
)

echo ============================================================
echo  PRODUCTION TRAINING - XAUUSD Direction Model (REAL EDGE)
echo  78,595 params (176 -^> 256 -^> 128 -^> 3)
echo  GPU: RTX 3050 cuBLAS ^| Batch 1024
echo  Features: z-score normalized, class-weighted CE loss
echo  Labels: 2.0x spread threshold, 256-tick horizon
echo  Validation: directional accuracy, confidence-gated edge
echo  Ctrl+C to stop - best model saved by EDGE metric
echo ============================================================
echo.
echo Building with GPU...
cmake --build build-cuda --config Release

if errorlevel 1 (
    echo.
    echo GPU build failed, falling back to CPU build...
    cmake --build build-vs --config Release
    if errorlevel 1 (
        echo BUILD FAILED
        pause
        exit /b 1
    )
    echo.
    build-vs\Release\from.exe train --data XAUUSD_ticks_all.parquet --lr 0.0003 --max-steps 999999999 --validate-every 5000 --save-every 50000 --batch-size 32 --max-samples 5000000
    exit /b 0
)

echo.

REM Production training for REAL TRADING EDGE:
REM - Z-score feature normalization (prevents scale dominance)
REM - Class-weighted CE loss (prevents majority-class collapse)
REM - Cosine LR annealing with warm restarts (escapes bad basins)
REM - 2.0x spread threshold (only labels REAL moves, not noise)
REM - 256-tick horizon (predicts further = more signal)
REM - Validates on: directional precision, confidence-gated win rate, expected edge
REM - Best model saved by EDGE metric (not just loss)
REM - Runs forever until Ctrl+C with periodic checkpoints
build-cuda\Release\from.exe train --data XAUUSD_ticks_all.parquet --lr 0.0003 --max-steps 999999999 --validate-every 5000 --save-every 50000 --batch-size 1024 --max-samples 5000000
