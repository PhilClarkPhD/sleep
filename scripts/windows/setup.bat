@echo off
REM One-time environment setup on Windows, AFTER Git, Python 3.11 and Node are installed.
REM Double-click this file, or run it from PowerShell. See docs\guides\SETUP_WINDOWS.md.
cd /d "%~dp0..\.."
echo === Creating Python environment in .venv ===
py -3.11 -m venv .venv 2>nul || python -m venv .venv
if errorlevel 1 goto fail
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 goto fail
echo === Running tests ===
pytest tests -q
echo === Installing GUI packages ===
cd frontend
call npm install
if errorlevel 1 goto fail
call npm run build
cd ..
echo.
echo Setup finished. Next: double-click scripts\windows\start_backend.bat and start_gui.bat
pause
exit /b 0
:fail
echo.
echo Something failed above. Copy the red text into your AI assistant along with docs\guides\SETUP_WINDOWS.md
pause
exit /b 1
