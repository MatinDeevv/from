@echo off
cd /d C:\Users\marti\from\from

set "PATH="
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=amd64 -host_arch=amd64 >NUL 2>&1

echo Building...
cmake --build build-vs --config Release

if errorlevel 1 (
    echo BUILD FAILED
    pause
    exit /b 1
)

echo.
echo === Backtesting latest checkpoint ===
echo.

REM Find the latest checkpoint
set "LATEST="
for %%f in (weights_step_*.from) do set "LATEST=%%f"

if "%LATEST%"=="" (
    if exist weights.from (
        set "LATEST=weights.from"
    ) else (
        echo No checkpoint found!
        pause
        exit /b 1
    )
)

echo Using: %LATEST%
echo.

build-vs\Release\from.exe backtest --model %LATEST% --data XAUUSD_ticks_all.parquet --ticks 5000000 --threshold 0.55

pause
