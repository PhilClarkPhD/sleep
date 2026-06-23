# Mora Sleep Scoring

Automated sleep state classification (Wake, NREM, REM) from EEG/EMG recordings in rats, using 10-second epochs.

## Architecture

- **Backend** (FastAPI): Loads an XGBoost model, accepts WAV uploads, returns sleep scores
- **Frontend** (React): Web UI for uploading, visualizing, editing, and exporting scores
- **CLI**: Command-line tool for local or API-based scoring
- **Model**: Training pipeline with time-series cross-validation and diagnostics

## Quick Start

### Web App (local)

```bash
# Terminal 1 - Backend
cd backend
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8000

# Terminal 2 - Frontend
cd frontend
npm install
npm run dev
# Opens at http://localhost:3000
```

### CLI

```bash
# Score locally
python cli/mora_cli.py score path/to/recording.wav -o scores.csv

# Score via API
python cli/mora_cli.py score path/to/recording.wav --api
```

### Deployed

- Frontend: https://sleep-cz3.pages.dev
- Backend: https://sleep-production.up.railway.app
- API Docs: https://sleep-production.up.railway.app/docs

## Repository Structure

```
backend/          FastAPI server + ML pipeline
frontend/         React + Plotly.js web UI
model/            Model training, cross-validation, diagnostics
cli/              Command-line interface
tests/            Unit and integration tests
sample_data/      Example WAV files for testing
model_artifacts/  Versioned model files
feature_store/    Pre-computed feature CSVs
training_data/    Raw EEG/EMG recordings (not in git)
```

## Background

Our lab studies sleep in the context of addiction and dopamine. We hand-scored EEG/EMG data in 10-second epochs — roughly 14+ hours per rat. This project automates that process using an XGBoost classifier, cutting analysis time to minutes.

The web interface (Mora) lets non-technical users upload recordings, view scored hypnograms alongside EEG/EMG signals, manually edit scores, and export results — all from a browser.
