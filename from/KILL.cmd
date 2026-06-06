@echo off
REM ========================================================================
REM KILL - Stop all training processes immediately
REM ========================================================================

echo.
echo ========================================================================
echo KILLING ALL FROM TRAINING PROCESSES
echo ========================================================================
echo.

tasklist /FI "IMAGENAME eq from.exe" 2>NUL | find /I /N "from.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo Found running from.exe processes - killing now...
    taskkill /F /IM from.exe
    echo.
    echo ========================================================================
    echo All from.exe processes killed!
    echo ========================================================================
) else (
    echo No from.exe processes found.
)

echo.
pause
