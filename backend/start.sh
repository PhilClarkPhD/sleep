#!/bin/sh
# Use Railway's PORT if set, otherwise default to 8000
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
