# Guide: first-time setup

Goal: a laptop that can run the GUI locally, run the tests, and retrain the model.
Do this once. Afterwards see `RUN_THE_APP.md`.

If you are an AI assistant walking someone through this: run one block at a time,
check its output, and stop at the first error. Every step says what success looks like.

## 0. What you need installed

| Tool | Version | Check | Install |
|---|---|---|---|
| git | any recent | `git --version` | https://git-scm.com |
| Python | 3.11 recommended (3.9+ works) | `python3 --version` | https://www.python.org/downloads/ |
| Node.js | 22.x (20+ works) | `node --version` | https://nodejs.org (LTS) |

Python 3.11 matches the production Docker image. The original developer used 3.9.5
via `asdf` (see `.tool-versions`); either is fine.

## 1. Get the code

```bash
git clone git@github.com:PhilClarkPhD/sleep.git
cd sleep
git checkout claude/refactor
```

`claude/refactor` is the branch that is deployed. `main` is older.

## 2. Python environment (backend, model, CLI, tests)

One virtual environment at the repo root covers everything:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Success looks like: `pip list` shows `fastapi`, `xgboost`, `pytest`.
The root `requirements.txt` pulls in `backend/requirements.txt` plus the CLI and
test dependencies.

Check the model loads and the tests pass:

```bash
pytest tests/ -q
```

Expect `88 passed` (the number grows as tests are added; what matters is no failures).

## 3. Node environment (GUI)

```bash
cd frontend
npm install
cd ..
```

Success looks like: a `frontend/node_modules/` folder exists and `npm run build`
(inside `frontend/`) ends with "built in N s".

## 4. Optional: data for retraining

Retraining needs hand-scored recordings, which are not in git (too large, lab data).
Get the `training_data/` folder from the lab's shared drive and put it at the repo
root so you have `training_data/Filtered Scores 2/ZZ05_BL/...`. The layout is
described in `docs/DATA_FORMATS.md`. You do not need this to run the GUI.

## 5. Optional: editor

VS Code with the Python and ESLint extensions is enough. There is a `.idea/` folder
from PyCharm; ignore it.

## Troubleshooting

- `ModuleNotFoundError: sleep_features` when running something in `backend/`: you
  ran it from the wrong folder, or the venv isn't active. The backend adds the repo
  root to `sys.path` automatically when started from `backend/`.
- `pip install` fails building `xgboost` or `scipy`: your Python is too new for the
  pinned wheels. Use Python 3.11.
- `npm install` warnings about peer dependencies are normal. Errors are not.
- On macOS, Phil's shell aliases `head` to something else. If a doc says `| head`,
  use `| sed -n '1,10p'`.
