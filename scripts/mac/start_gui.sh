#!/bin/bash
# Starts the Mora GUI on http://localhost:3000. Start the backend first. Leave this terminal open.
cd "$(dirname "$0")/../../frontend"
if [ ! -d node_modules ]; then
  echo "node_modules not found. Run: bash scripts/mac/setup.sh"
  exit 1
fi
echo "Starting GUI on http://localhost:3000  (Ctrl+C to stop)"
exec npm run dev
