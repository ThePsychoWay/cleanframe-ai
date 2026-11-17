@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (
  py run_local.py %*
  goto finished
)
where python >nul 2>nul
if %errorlevel%==0 (
  python run_local.py %*
  goto finished
)
echo.
echo   Python 3.11 or newer was not found.
echo   Install it from https://www.python.org/downloads/ (tick "Add Python to PATH"), then double-click this file again.
:finished
echo.
pause
