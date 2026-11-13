@echo off
REM Starts the Mora API on http://localhost:8000. Leave this window open while using the GUI.
cd /d "%~dp0..\.."
if not exist .venv\Scripts\activate.bat (
  echo No .venv found. Run scripts\windows\setup.bat first.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
cd backend
echo Starting backend on http://localhost:8000  (Ctrl+C to stop)
uvicorn app.main:app --reload --port 8000
pause
