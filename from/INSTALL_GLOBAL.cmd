@echo off
REM ========================================================================
REM Install FROM globally - run "from" from anywhere
REM ========================================================================

echo.
echo ========================================================================
echo Installing FROM globally
echo ========================================================================
echo.

set INSTALL_DIR=%USERPROFILE%\bin
set FROM_DIR=%~dp0

echo Creating install directory: %INSTALL_DIR%
if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"

echo Creating wrapper script: %INSTALL_DIR%\from.cmd
(
echo @echo off
echo cd /d "%FROM_DIR%"
echo if not exist build-vs\Release\from.exe ^(
echo     echo FROM not built yet. Building...
echo     call build_maximum_speed.cmd
echo ^)
echo build-vs\Release\from.exe %%*
) > "%INSTALL_DIR%\from.cmd"

REM Check if already in PATH
echo %PATH% | findstr /C:"%INSTALL_DIR%" >nul
if errorlevel 1 (
    echo.
    echo ========================================================================
    echo IMPORTANT: Add to PATH manually
    echo ========================================================================
    echo.
    echo Run this command in PowerShell ^(as Administrator^):
    echo.
    echo [Environment]::SetEnvironmentVariable^("Path", [Environment]::GetEnvironmentVariable^("Path", "User"^) + ";%INSTALL_DIR%", "User"^)
    echo.
    echo Then restart your terminal.
    echo.
    echo ========================================================================
) else (
    echo %INSTALL_DIR% already in PATH - you're good!
)

echo.
echo ========================================================================
echo Installation complete!
echo ========================================================================
echo.
echo You can now run:
echo   from train-fast --data mydata.parquet
echo.
echo From any directory!
echo ========================================================================

pause
