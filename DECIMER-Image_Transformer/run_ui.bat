@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo [DECIMER] Creating Python environment...
  python -m venv .venv
  if errorlevel 1 (
    echo Python 3.9 or newer was not found. Install 64-bit Python, then run this file again.
    pause
    exit /b 1
  )
)
echo [DECIMER] Checking dependencies...
".venv\Scripts\python.exe" -m pip install -r requirements-ui.txt
if errorlevel 1 (
  echo Dependency installation failed. Check the network and try again.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" decimer_desktop.py
endlocal
