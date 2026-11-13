#!/bin/bash
# One-time environment setup on macOS/Linux, AFTER git, Python 3.11 and Node are installed.
# Run:  bash scripts/mac/setup.sh     (from anywhere)
set -e
cd "$(dirname "$0")/../.."
echo "=== Creating Python environment in .venv ==="
PY=python3
command -v python3.11 >/dev/null 2>&1 && PY=python3.11
$PY -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo "=== Running tests ==="
pytest tests -q
echo "=== Installing GUI packages ==="
cd frontend
npm install
npm run build
cd ..
echo
echo "Setup finished. Next: bash scripts/mac/start_backend.sh and bash scripts/mac/start_gui.sh"
