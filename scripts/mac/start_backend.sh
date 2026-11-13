#!/bin/bash
# Starts the Mora API on http://localhost:8000. Leave this terminal open while using the GUI.
cd "$(dirname "$0")/../.."
if [ ! -f .venv/bin/activate ]; then
  echo "No .venv found. Run: bash scripts/mac/setup.sh"
  exit 1
fi
source .venv/bin/activate
cd backend
echo "Starting backend on http://localhost:8000  (Ctrl+C to stop)"
exec uvicorn app.main:app --reload --port 8000
