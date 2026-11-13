# Project brief for AI assistants

This repo is **Mora**: automated sleep-state scoring (Wake / Non REM / REM, 10 s
epochs) for rat EEG/EMG recordings, plus a web app to review and correct the scores.
Treat the user as a neuroscientist who may not be a programmer. Explain web and
deployment concepts briefly before using them. Run one command at a time and check
the output.

## Map

- `sleep_features/`: feature engineering, the single source of truth for WAV → per-epoch features. Used by both training and serving.
- `model/`: `train.py`, `evaluate.py`, diagnostics, CV. `model/experiments/`: research scripts, not production.
- `backend/`: FastAPI server; serves the `.pkl` in `backend/models/`. `app/services/scoring_service.py` is the pipeline.
- `frontend/`: React + TypeScript GUI. One page, one Zustand store, one API module.
- `cli/mora_cli.py`: command-line scoring, same code path as the backend.
- `docs/`: **start at `README.md`'s "I want to..." table**, which maps tasks to guides in `docs/guides/`.

## Facts that are easy to get wrong

- Everything runs locally (backend on :8000, GUI on :3000). Working branch is `claude/refactor`. The hosted Railway/Cloudflare copy was shut down in Nov 2026; `docs/guides/DEPLOY.md` is only for re-hosting.
- The backend serves whichever `.pkl` is in `backend/models/` (newest if several). Promote a model by copying its `.pkl` and `_ood.pkl` there.
- A model's metadata (feature columns, normalization strategy, feature-code version, OOD file) tells the backend how to featurize. Never hardcode feature lists in serving code.
- Default scores are **unfiltered**. The rule-based filter is a UI toggle only. Single-epoch REM counts as real in this lab.
- Training labels are the human-corrected files; `_scores.csv` exports next to them are old model output. `labels.json` can override or exclude (`null`) a recording.
- Reliability (QC %, EMG range, OOD %) is a recording-level warning, not per-epoch blanking. Only signal QC marks epochs `Unscored`.
- `FEATURE_CODE_VERSION` in `sleep_features/config.py` must be bumped when feature math changes (MAJOR) or columns are added (MINOR).
- `training_data/`, `feature_store/`, `model_cards/` are not in git.
- Run `pytest -q` (unit + integration) and `cd frontend && npm run build` before calling a change done.
- The original developer's shell aliases `head`; use `sed -n '1,20p'` in commands you suggest.

## Decisions already made (don't reopen without new evidence)

See `docs/MODEL.md`, "Decisions already made". Open backlog is in the same file.
