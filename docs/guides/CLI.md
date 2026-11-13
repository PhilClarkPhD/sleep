# Guide: score from the command line

For scoring many files, scripting, or checking a model without the GUI.
Prerequisite: `SETUP.md` or `SETUP_WINDOWS.md` (the root `requirements.txt` includes
the CLI deps). Activate the venv first; on Windows use `python cli\mora_cli.py`.

## Two modes

| Mode | Command | Runs where | Needs |
|---|---|---|---|
| Local | `python cli/mora_cli.py score file.wav` | your machine, the `.pkl` in `backend/models/` | Python env from SETUP |
| API | `python cli/mora_cli.py score file.wav --api` | a running backend, `http://localhost:8000` by default | backend started (`RUN_THE_APP.md`); `--api-key` only if auth is on |

Both run the **same scoring code** as the web app (`backend/app/services/scoring_service.py`),
so results match the GUI: unfiltered scores, signal-QC `Unscored` epochs, and the
recording-level reliability warnings.

Optional: `pip install -e ./cli` gives you a `mora` command instead of
`python cli/mora_cli.py`.

## Commands

### `score`: one file

```bash
python cli/mora_cli.py score recording.wav -o recording_scores.csv
python cli/mora_cli.py score recording.wav --api
python cli/mora_cli.py score recording.wav --api --api-key YOUR_KEY
python cli/mora_cli.py score recording.wav --json > result.json
```

Options:

- `-o, --output PATH`: write `epoch,score` CSV (same first two columns as the GUI export).
- `--api`: use the remote server instead of the local model.
- `--api-url URL`: which backend (default `http://localhost:8000`; a hosted URL if you have one).
- `--api-key KEY`: sent as `X-API-Key`. Or set the environment variable `MORA_API_KEY`.
- `--start-epoch N`: baseline epoch for the legacy model (default 2, same as the API). Ignored by `robust_mad` models.
- `--json`: machine-readable output: `{"scores": [...], "summary": {...}}`.

Output (table mode):

```
Scoring: recording.wav
  Loading model from .../backend/models/XGBoost_1.2.4.pkl
  Scoring...

Summary:
----------------------------------------
State          Epochs       Time        %
----------------------------------------
Wake             3900    650.0m    45.1%
Non REM          4050    675.0m    46.9%
REM               690    115.0m     8.0%
----------------------------------------
Recording quality: OK (QC fail 0.02%)
```

or, when a reliability check trips:

```
WARNING: this recording may be harder to score. Review results carefully.
  - [weak_emg_contrast] EMG range p05/p50 = 0.907 (threshold 0.6): ...
```

Percentages are over scorable epochs; an `Unscored` row appears if any epoch
failed signal QC.

### `batch`: a folder

```bash
python cli/mora_cli.py batch path/to/wavs/ -o path/to/output/
python cli/mora_cli.py batch path/to/wavs/ -o out/ --api --api-key YOUR_KEY
```

Scores every `*.wav` / `*.WAV` in the folder (not recursive), writes
`<name>_scores.csv` per file into the output folder, prints each file's reliability
line, and keeps going if one file fails.

### `info`: what the server is running

```bash
python cli/mora_cli.py info [--api-url URL] [--api-key KEY]
```

Prints model name, version, classes, and feature columns from a running backend
(local by default). Same information as http://localhost:8000/api/v1/model/info.

## Scripting

`--json` gives `summary["Wake"]["count"|"percentage"|"minutes"]` etc., plus
`summary["reliability"]` (the same block the API returns) and
`summary["model_version"]`. The CSV from `-o` is two columns, `epoch` from 0.

## Choosing the local model

The local mode loads whatever the backend would: the `.pkl` in `backend/models/`,
or `MORA_MODEL_PATH` if set:

```bash
MORA_MODEL_PATH=model_artifacts/XGBoost_1.2.3/XGBoost_1.2.3.pkl python cli/mora_cli.py score file.wav
```

Windows: `$env:MORA_MODEL_PATH = "model_artifacts\XGBoost_1.2.3\XGBoost_1.2.3.pkl"` on its own line first.

## Troubleshooting

- `Local scoring needs the backend dependencies`: activate the venv and
  `pip install -r requirements.txt` from the repo root, or use `--api`.
- `Connection refused` with `--api`: the backend isn't running. Start it (`RUN_THE_APP.md`) or drop `--api` to score in-process.
- `The API requires a key`: pass `--api-key` or set `MORA_API_KEY`.
- `Expected .wav file`: the file extension must be `.wav`.
- Very slow first run locally: XGBoost and SciPy import time; subsequent runs are faster.
