"""Mora backend application package.

Bootstrap: make the repo-root `sleep_features` package importable regardless of
how the backend is launched (uvicorn from backend/, tests from repo root, or the
Docker image where /app holds both `app` and `sleep_features`).
"""

import sys as _sys
from pathlib import Path as _Path

_repo_root = _Path(__file__).resolve().parents[2]
if (_repo_root / "sleep_features").is_dir() and str(_repo_root) not in _sys.path:
    _sys.path.insert(0, str(_repo_root))
