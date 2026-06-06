@echo off
REM Build with CUDA support for maximum GPU utilization

echo ========================================================================
echo Building FROM with CUDA support
echo This will enable real GPU acceleration with massive batch sizes
echo ========================================================================

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=amd64 -host_arch=amd64

cd /d %~dp0

REM Clean and rebuild with CUDA
if exist build-cuda rmdir /s /q build-cuda
cmake -S . -B build-cuda -G "NMake Makefiles" ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DFROM_CUDA=ON ^
  -DFROM_NATIVE_ARCH=ON ^
  -DCMAKE_CXX_COMPILER="C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe" ^
  -DCMAKE_C_COMPILER="C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe"

if errorlevel 1 (
    echo.
    echo CUDA not found! Building without CUDA...
    cmake -S . -B build-cuda -G "NMake Makefiles" ^
      -DCMAKE_BUILD_TYPE=Release ^
      -DFROM_NATIVE_ARCH=ON ^
      -DCMAKE_CXX_COMPILER="C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe" ^
      -DCMAKE_C_COMPILER="C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe"
)

cmake --build build-cuda --config Release -j 8

if errorlevel 1 (
    echo Build failed!
    exit /b 1
)

echo.
echo ========================================================================
echo Build complete!
echo Run with: build-cuda\from.exe train --data XAUUSD_ticks_all.parquet
echo Batch size is now 4096 (was 256) to maximize GPU utilization
echo ========================================================================
