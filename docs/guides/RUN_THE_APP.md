# Guide: run the GUI locally and score a recording

Prerequisite: `SETUP.md` done. You need two terminal windows: one for the Python
backend, one for the frontend. Both stay running while you use the app.

## 1. Start the backend (API)

```bash
cd sleep                      # repo root
source .venv/bin/activate
cd backend
uvicorn app.main:app --reload --port 8000
```

Success looks like a line containing `Model loaded successfully: 1.2.4` (version
may differ) followed by `Uvicorn running on http://127.0.0.1:8000`.

Check it in a browser: http://localhost:8000/api/v1/health should show
`"status":"healthy","model_loaded":true`. http://localhost:8000/docs is an
interactive page listing every endpoint.

If it says `degraded` / `model_loaded: false`, there is no `.pkl` in
`backend/models/`. See "Which model is served" below.

## 2. Start the frontend (GUI)

In a second terminal:

```bash
cd sleep/frontend
npm run dev
```

Success looks like `Local: http://localhost:3000/`. Open that in Chrome or Edge
(best; Safari and Firefox work but lack the "choose export folder" feature).

The page header should show "Model: XGBoost v1.2.4" (or similar). If it shows
"Connecting to API..." forever, the backend isn't running on port 8000.

## 3. Score a recording

1. Drag a stereo WAV (channel 1 EEG, channel 2 EMG) onto the upload box, or click to
   browse. `sample_data/sample_signal.wav` is a 90-minute test file.
2. Leave "Baseline epoch" at 2. It only matters for the older model; newer models
   ignore it.
3. Wait for the progress bar. A 24-hour file takes about 30 to 60 seconds.
4. You get: the epoch viewer (EEG, EMG, power spectrum), the hypnogram, the summary
   with counts and percentages, Recording Settings, and the Export panel.

If an amber box appears under "Scoring Results" saying "Review recommended", the
recording tripped a reliability check (noisy, weak EMG contrast, or unlike the
training data). Scores are still produced; check them more carefully. See
`docs/MODEL.md`, "Reliability warnings".

## 4. Review and correct

- **Move**: ← / → keys, Prev/Next buttons, click anywhere on the hypnogram,
  Ctrl+F (Cmd+F) to jump to an epoch number, "Next REM" button.
- **Re-score the current epoch**: **W** Wake, **E** Non REM, **R** REM, **T** Unscored.
  Each key saves and advances to the next epoch. The same four buttons exist on screen.
- **Window**: dropdown sets how many epochs are visible around the current one (1 to 20).
- **Y-axes**: each plot autoscales per epoch by default. "Lock Y" freezes the range
  so you can compare epochs; "Reset Y-axes" clears all locks.
- **Post-processing filter toggle** (in Scoring Results): shows the model's raw
  output (default) or a smoothed version that removes isolated single epochs. The
  lab convention is that single-epoch REM is real, so the raw view is the default.

Nothing is saved until you export. **Reloading the page loses everything.**

## 5. Recording settings (optional, for light/dark breakdown)

Set the recording start date and time, then add light/dark phases or click a 12h
preset (lights on 6am / off 6pm, etc.). This adds a clock-time axis to the
hypnogram and a `phase` column plus per-phase totals to the exports.

## 6. Export

- **Choose folder…** (Chrome/Edge): pick a folder once; every export goes straight
  there and overwrites files with the same name.
- **Export Scores (CSV)**: `<recording>_scores.csv` with columns
  `epoch,score,timestamp_seconds[,datetime][,phase]`.
- **Export Breakdown**: `<recording>_breakdown.csv` with stage counts and
  proportions, overall and per light/dark phase.
- **Import Scores**: load a previously exported (or hand-scored) CSV to continue
  editing. Needs `epoch` and `score` columns.

## Which model is served

The backend serves the single `*.pkl` in `backend/models/` (ignoring `*_ood.pkl`).
To serve a different model, put its `.pkl` and `_ood.pkl` there, delete the old one,
restart the backend. To point somewhere else without copying:

```bash
MORA_MODEL_PATH=../model_artifacts/XGBoost_1.2.3/XGBoost_1.2.3.pkl uvicorn app.main:app --port 8000
```

## Stopping

Ctrl+C in each terminal.

## Is there an online version?

Not any more. The hosted copy was shut down in November 2026. Running locally is
the normal way to use the app. `DEPLOY.md` explains how to host it again if needed.
