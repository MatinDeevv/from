@echo off
REM ========================================================================
REM BUILD FOR MAXIMUM SPEED - 80%% GPU, 80%% CPU, 80%% RAM
REM ========================================================================

echo.
echo ========================================================================
echo BUILDING FROM - MAXIMUM PERFORMANCE MODE
echo ========================================================================
echo.
echo Target utilization:
echo   - GPU:  80%% utilization + 80%% VRAM (batch_size=8192)
echo   - CPU:  80%% cores for async data loading (%NUMBER_OF_PROCESSORS% threads)
echo   - RAM:  80%% for data prefetch pipeline (24 batches = ~3GB)
echo.
echo Features:
echo   - Async multi-threaded data loader (fills CPU)
echo   - Triple-buffered GPU pipeline (upload/compute/download overlap)
echo   - Massive batch size (8192 samples = 2.5GB VRAM)
echo   - Prefetch 24 batches in RAM (workers load while GPU computes)
echo.
echo ========================================================================

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=amd64 -host_arch=amd64

cd /d %~dp0

REM Try CUDA first (fastest)
echo.
echo [1/3] Attempting CUDA build (fastest)...
if exist build-max rmdir /s /q build-max
cmake -S . -B build-max -G "NMake Makefiles" ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DFROM_CUDA=ON ^
  -DFROM_NATIVE_ARCH=ON ^
  -DCMAKE_CXX_COMPILER="C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe" ^
  -DCMAKE_C_COMPILER="C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe"

if errorlevel 1 (
    echo CUDA not found. Falling back to D3D11 DirectCompute...
    cmake -S . -B build-max -G "NMake Makefiles" ^
      -DCMAKE_BUILD_TYPE=Release ^
      -DFROM_NATIVE_ARCH=ON ^
      -DCMAKE_CXX_COMPILER="C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe" ^
      -DCMAKE_C_COMPILER="C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe"
)

echo.
echo [2/3] Building with maximum optimization...
cmake --build build-max --config Release -j %NUMBER_OF_PROCESSORS%

if errorlevel 1 (
    echo.
    echo ========================================================================
    echo BUILD FAILED
    echo ========================================================================
    exit /b 1
)

echo.
echo [3/3] Build complete!
echo.
echo ========================================================================
echo SUCCESS - Maximum performance build ready
echo ========================================================================
echo.
echo Executable: build-max\from.exe
echo.
echo Run with:
echo   build-max\from.exe train-fast --data XAUUSD_ticks_all.parquet --max-steps 1000
echo.
echo Expected performance:
echo   - Old:  5-10 steps/minute   (batch=256, sync loading)
echo   - New:  200-500 steps/minute (batch=8192, async pipeline)
echo   - Speedup: 40-100x faster!
echo.
echo Resource usage:
echo   - GPU:  80-95%% utilization, 3-4GB VRAM
echo   - CPU:  70-90%% (all cores loading data in parallel)
echo   - RAM:  12-14GB (prefetch pipeline + OS + model)
echo.
echo ========================================================================
