# Mora Sleep Scoring

Automated sleep-state scoring (Wake, Non REM, REM) for rat EEG/EMG recordings in
10-second epochs. A trained XGBoost model does the first pass; a web app lets you
check, correct, and export the scores.

Three parts, all in this repo:

| Part | What it is | Where |
|---|---|---|
| **Web app (GUI)** | Browser page: upload a WAV, see the hypnogram and signals, fix scores, export CSV | `frontend/` |
| **API (backend)** | Python server that runs the model. The GUI and the CLI both call it | `backend/` |
| **Model + features** | Feature engineering (`sleep_features/`), training and evaluation (`model/`) | `sleep_features/`, `model/` |

Deployed copies: GUI at https://sleep-cz3.pages.dev, API at
https://sleep-production.up.railway.app (interactive docs at `/docs`).

---

## I want to ... (start here)

Each guide is written to be handed to an AI assistant along with your question.
Open the file, paste it (or point your assistant at it), and say what you want to do.

| I want to... | Read this | Needs |
|---|---|---|
| Set up my computer for the first time | [docs/guides/SETUP.md](docs/guides/SETUP.md) | Python, Node |
| Run the GUI on my own machine and score a recording | [docs/guides/RUN_THE_APP.md](docs/guides/RUN_THE_APP.md) | Setup done |
| Understand how the pieces fit together | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Nothing |
| Retrain the model (new recordings, new labels) | [docs/guides/RETRAIN_MODEL.md](docs/guides/RETRAIN_MODEL.md) | Setup, labeled data |
| Evaluate a model / compare two models | [docs/guides/EVALUATE_MODEL.md](docs/guides/EVALUATE_MODEL.md) | A trained model |
| Try a new feature (input to the model) | [docs/guides/ADD_A_FEATURE.md](docs/guides/ADD_A_FEATURE.md) | Retrain guide first |
| Change something in the GUI | [docs/guides/EDIT_THE_UI.md](docs/guides/EDIT_THE_UI.md) | Node |
| Understand the GUI's design and behavior | [docs/GUI_DESIGN.md](docs/GUI_DESIGN.md) | Nothing |
| Understand the model and its known limits | [docs/MODEL.md](docs/MODEL.md) | Nothing |
| Score files from the command line / in bulk | [docs/guides/CLI.md](docs/guides/CLI.md) | Setup |
| Run the tests | [docs/guides/TESTING.md](docs/guides/TESTING.md) | Setup |
| Put a new model or UI change online | [docs/guides/DEPLOY.md](docs/guides/DEPLOY.md) | Railway / Cloudflare access |
| Know what files the pipeline expects | [docs/DATA_FORMATS.md](docs/DATA_FORMATS.md) | Nothing |
| Turn on API keys / rate limits | [docs/SECURITY_GUIDE.md](docs/SECURITY_GUIDE.md) | Railway access |

If you are an AI assistant reading this: `CLAUDE.md` at the repo root is the short
project brief. Read it, then the guide matching the user's task.

---

## Quick start (after setup)

```bash
# Terminal 1 - API
cd backend
source ../.venv/bin/activate
uvicorn app.main:app --reload --port 8000

# Terminal 2 - GUI
cd frontend
npm run dev
# open http://localhost:3000
```

## Repository map

```
backend/            FastAPI server. app/ is the code, models/ holds the served model .pkl
frontend/           React + TypeScript web app (Vite, Tailwind, Plotly)
sleep_features/     Feature engineering package: WAV -> per-epoch feature table (+ QC, OOD, reliability)
model/              train.py, evaluate.py, diagnostics, cross-validation, model_config.json
model/experiments/  Research scripts behind the v2 feature decisions (not used in production)
cli/                mora_cli.py: score files locally or via the API
tests/              pytest suite (unit + integration + golden sample)
docs/               All documentation (guides/ = task walkthroughs; archive/ = history)
model_artifacts/    Every trained model, one folder per version (committed)
feature_store/      Feature tables built from training_data (NOT in git)
training_data/      Raw WAVs + hand-scored labels (NOT in git; see docs/DATA_FORMATS.md)
model_cards/        HTML reports per model version (NOT in git)
sample_data/        A tiny WAV + scores for smoke tests
notebooks/          Legacy exploration notebooks (stale paths; kept for reference)
```

## Background

Our lab studies sleep in the context of addiction and dopamine. Hand-scoring a
24-hour EEG/EMG recording in 10-second epochs takes a trained person most of a day.
This project automates the first pass with an XGBoost classifier trained on
hand-scored recordings, and gives non-programmers a browser tool to review and
correct the result. Current model agreement with human scorers is about kappa 0.84;
REM is the hardest state (recall about 0.8). See [docs/MODEL.md](docs/MODEL.md).
