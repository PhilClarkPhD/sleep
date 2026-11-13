# Architecture

How the pieces fit together, what talks to what, and where the boundaries are.
No prior web-development knowledge assumed.

## One-paragraph version

A **recording** is a stereo WAV file: channel 1 is EEG, channel 2 is EMG, usually
1000 samples per second for about 24 hours. The **feature package**
(`sleep_features/`) cuts it into 10-second epochs and computes a handful of numbers
per epoch (EEG amplitude, EMG amplitude, delta and theta power, and so on). The
**model** (an XGBoost classifier in a `.pkl` file) turns each epoch's numbers into
Wake / Non REM / REM. The **backend** (`backend/`, Python, FastAPI) is a small web
server that wraps those two steps behind an HTTP endpoint. The **frontend**
(`frontend/`, React) is the page you open in a browser: it uploads the WAV to the
backend, draws the result, lets you edit scores, and saves a CSV. The **CLI**
(`cli/`) does the same scoring without the browser.

## Diagram

```
                 browser (frontend/)                    python server (backend/)
  ┌─────────────────────────────────────┐   HTTP    ┌─────────────────────────────────┐
  │  Upload WAV  → progress bar         │ ───────▶  │ POST /api/v1/score              │
  │  Hypnogram, EEG/EMG plots           │ ◀───────  │   load WAV                      │
  │  Keyboard re-scoring (W/E/R/T)      │  JSON     │   sleep_features.generate_...   │  ← sleep_features/
  │  Export CSV / breakdown             │           │   model.predict (XGBoost .pkl)  │  ← backend/models/*.pkl
  └─────────────────────────────────────┘           │   signal QC → "Unscored"        │
                                                    │   reliability warnings          │
                 terminal (cli/)                     │ GET /api/v1/health, /model/info │
  ┌─────────────────────────────────────┐           └─────────────────────────────────┘
  │ mora score file.wav [--api]         │ ── same code path (ScoringService) ──┘
  └─────────────────────────────────────┘

                 training (model/, offline, your laptop)
  training_data/*.wav + labels ──▶ python -m sleep_features build ──▶ feature_store/*.csv
  feature_store/*.csv ──▶ python -m model.train ──▶ model_artifacts/<id>/<id>.pkl (+ _ood.pkl)
  copy the .pkl files into backend/models/ ──▶ the backend serves the new model
```

## The components

### `sleep_features/` : feature engineering (single source of truth)

Pure Python package, no web code. Used by the backend at scoring time **and** by the
feature-store builder at training time, so the numbers the model sees in training
are computed by exactly the same code as the numbers it sees in production.

- `pipeline.generate_feature_table(df, normalization, samplerate, feature_set)` is the
  one entry point. Input: a DataFrame with `eeg` and `emg` columns. Output: one row
  per 10-second epoch.
- `transforms.py`, `spectral.py`, `temporal.py`: the math (FFT band power, amplitude
  stats, Hjorth parameters, rolling context).
- `normalize.py`: per-recording normalization. `robust_mad` (default: median/MAD
  per recording) replaces the old `baseline_epoch` method, which needed a user to
  pick a "known Wake epoch".
- `qc.py`: per-epoch signal-quality checks (flat line, clipping, non-finite, gross
  amplitude). Epochs that fail are reported as **Unscored**.
- `ood.py` + `reliability.py`: recording-level warnings. See `docs/MODEL.md`,
  section "Reliability warnings".
- `discovery.py`, `scores.py`, `io.py`, `build.py`: find recordings and label files
  on disk, parse labels, build the feature store CSV + manifest.
- `config.py`: feature column lists (v1, v2), bands, epoch length, and
  `FEATURE_CODE_VERSION`, the compatibility contract between a saved model and the
  feature code (see `docs/guides/ADD_A_FEATURE.md`).

### `model/` : training and evaluation (offline)

- `train.py`: reads `model_config.json`, loads a feature-store CSV, splits by
  recording and time, tunes XGBoost with time-series cross-validation, reports
  holdout metrics, refits on everything, fits the OOD detector, saves to
  `model_artifacts/<model_id>/`.
- `evaluate.py`: load a saved model + a labeled feature CSV, print kappa, per-class
  precision/recall, confusion matrix, transition accuracy.
- `diagnostics.py`, `cross_validation.py`: the metrics and splitters both of those use.
- `experiments/`: research scripts (leave-one-animal-out comparisons, per-animal
  diagnosis). Not part of the product.

### `backend/` : the API

FastAPI app. Starts, loads the model (`app/core/model_loader.py`), and exposes:

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/v1/score` | Upload a WAV, get scores, summary, reliability warnings, and (optionally) downsampled signals for plotting |
| POST | `/api/v1/features` | Feature table only (debugging) |
| GET | `/api/v1/model/info` | Model name, version, feature columns, classes |
| GET | `/api/v1/health` | Is it up, is the model loaded, is an API key required |

`app/services/scoring_service.py` is the pipeline: load WAV, features (strategy chosen
by the model's metadata), predict, rule-based filter (kept for the UI toggle; the
default view is unfiltered), QC → Unscored, reliability report.

Configuration is by environment variables prefixed `MORA_` (`app/core/config.py`).
The important ones: `MORA_MODEL_PATH` (default: the `.pkl` in `backend/models/`),
`MORA_API_KEYS` (empty = no auth), `MORA_CORS_ORIGINS`.

The model `.pkl` is a tuple `(xgboost_model, metadata_dict, label_encoder)`. The
metadata tells the backend which feature columns, which normalization strategy, which
feature-code version, and where the OOD detector file is. **That is why you never
need to edit backend code to serve a new model**: drop the new `.pkl` (and its
`_ood.pkl`) in `backend/models/`, remove the old one, restart.

### `frontend/` : the GUI

React + TypeScript, built with Vite, styled with Tailwind, charts with Plotly.
One page. State lives in a single store (`src/store/useAppStore.ts`). The page
talks to the backend through `src/api/sleepApi.ts` only. Full description in
`docs/GUI_DESIGN.md`; how to change it in `docs/guides/EDIT_THE_UI.md`.

In development, the Vite dev server (port 3000) forwards `/api` requests to the
backend on port 8000, so no configuration is needed. In production the build is
given `VITE_API_URL` pointing at the Railway backend.

### `cli/` : command line

`mora score file.wav` runs the exact same `ScoringService` as the backend, in-process.
`mora score file.wav --api` uploads to the deployed backend instead. `mora batch` does
a folder. See `docs/guides/CLI.md`.

## Key concepts for newcomers

**Why a separate frontend and backend?** The model needs Python and 100+ MB of
libraries. A browser can't run that. So the Python lives on a server, and the browser
just sends a file and receives JSON. The user needs nothing installed.

**What is an API?** A set of URLs the server answers. `POST /api/v1/score` means
"send me a file at this address and I'll reply with scores". Open
`http://localhost:8000/docs` while the backend runs to see and try every endpoint.

**What is Docker?** A recipe (`backend/Dockerfile`) that builds a self-contained
Linux image with Python, the libraries, the code, and the model. Railway builds and
runs that image. You don't need Docker locally; the guides run things directly.

**Where does the model's "truth" come from?** `training_data/` (not in git): WAVs
plus hand-scored label files. `docs/DATA_FORMATS.md` describes the layout.

## Build time vs run time

- **Training** happens on a laptop, by hand, occasionally. Output: a `.pkl`.
- **Deploying** happens when you push to GitHub: Railway rebuilds the backend
  image (with whatever `.pkl` is in `backend/models/`), Cloudflare Pages rebuilds
  the frontend. Minutes.
- **Scoring** happens every time someone uploads a file. Seconds to a minute for a
  24-hour recording.

## Deployment (summary)

| What | Where | Config |
|---|---|---|
| Backend | Railway, Docker image from `backend/Dockerfile`, build context = repo root (`railway.json`) | env vars `MORA_*`, `PORT` set by Railway |
| Frontend | Cloudflare Pages project `sleep-cz3`, builds `frontend/` with `npm run build` | env var `VITE_API_URL` |
| Trigger | `git push` to branch `claude/refactor` (the production branch) | |

Details and the exact dashboard settings: `docs/guides/DEPLOY.md`.

## Security model

Optional. With `MORA_API_KEYS` empty (current state), anyone with the URL can score
files. Set it to enable the `X-API-Key` header check; the GUI has a gear icon to
enter the key, the CLI takes `--api-key`. Rate limiting and request logging are
also available. `docs/SECURITY_GUIDE.md` explains when you'd want this.
