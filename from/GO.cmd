@echo off
cd /d %~dp0

tasklist /FI "IMAGENAME eq from.exe" 2>NUL | find /I "from.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo Killing running from.exe...
    taskkill /F /IM from.exe >NUL 2>&1
    timeout /t 1 /nobreak >NUL
)

if not exist build-vs\Release\from.exe (
    echo Rebuilding...
    call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=amd64 >nul 2>&1
    cmake --build build-vs --config Release
    if errorlevel 1 exit /b 1
)

build-vs\Release\from.exe train-fast --data XAUUSD_ticks_all.parquet --max-steps 1000 --no-ui
