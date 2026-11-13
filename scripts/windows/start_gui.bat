@echo off
REM Starts the Mora GUI on http://localhost:3000. Start the backend first. Leave this window open.
cd /d "%~dp0..\..\frontend"
if not exist node_modules (
  echo node_modules not found. Run scripts\windows\setup.bat first.
  pause
  exit /b 1
)
echo Starting GUI on http://localhost:3000  (Ctrl+C to stop)
call npm run dev
pause
