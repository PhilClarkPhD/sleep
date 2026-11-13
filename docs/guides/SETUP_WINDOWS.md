# Guide: first-time setup on Windows

Goal: a Windows laptop with nothing installed becomes one that can run the GUI,
run the tests, and retrain the model. About 30 minutes, mostly installers. Do this
once; afterwards see `RUN_THE_APP.md`.

If you are an AI assistant walking someone through this: they are on Windows 10
or 11, using **PowerShell** (not Command Prompt, not WSL). Give one step at a time,
ask for the output, and stop at the first error. Every step says what success
looks like. Commands in other guides written for Mac (`source .venv/bin/activate`,
`python3`, `sed`) have Windows equivalents in the cheat sheet at the bottom.

## 1. Install three programs

Download and run each installer with the defaults, **except where noted**.

| Program | Where | Important checkbox |
|---|---|---|
| Git for Windows | https://git-scm.com/download/win | defaults are fine |
| Python 3.11 | https://www.python.org/downloads/windows/ → "Windows installer (64-bit)" for the latest **3.11.x** | On the first screen tick **"Add python.exe to PATH"** before clicking Install |
| Node.js 22 LTS | https://nodejs.org → the LTS button | defaults are fine |

Python 3.11 specifically: newer versions may not have prebuilt packages for
everything we use. If you already have another Python, 3.11 can be installed
alongside it.

After the installers, **close and reopen PowerShell** (Start menu → type
"PowerShell" → Enter), then check:

```powershell
git --version
python --version
node --version
npm --version
```

Success looks like: four version lines, Python saying `3.11.x`. If `python`
opens the Microsoft Store instead, the PATH box was not ticked: rerun the Python
installer, choose Modify, and tick it, or use `py -3.11` wherever a guide says
`python`.

## 2. Allow PowerShell to activate the Python environment

One-time, so the activation script in step 4 is allowed to run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Answer `Y`. This only affects your user account and only allows local scripts.

## 3. Get the code

Pick a folder with no spaces in the path, for example `C:\Users\<you>\code`.

```powershell
mkdir $HOME\code
cd $HOME\code
git clone https://github.com/PhilClarkPhD/sleep.git
cd sleep
git checkout claude/refactor
```

Success looks like: `cd sleep` works and `dir` shows `backend`, `frontend`,
`model`, `docs`. (`claude/refactor` is the current working branch; `main` is older.)

If GitHub asks you to log in, a browser window opens; sign in with the GitHub
account that has been given access to the repo.

## 4. Python environment (backend, model, CLI, tests)

Still in the `sleep` folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Success looks like: the prompt now starts with `(.venv)`, and the install ends
with `Successfully installed ...` listing `fastapi`, `xgboost`, `pytest`, and many
others. This step downloads a few hundred MB and takes 2 to 5 minutes.

Prove it works:

```powershell
pytest tests -q
```

Expect `88 passed` (the number grows as tests are added; what matters is no
failures). A Windows Firewall prompt may appear the first time; click Allow.

## 5. Node environment (GUI)

```powershell
cd frontend
npm install
npm run build
cd ..
```

Success looks like: `npm run build` ends with `built in N s`. Warnings about
"peer dependencies" or "funding" are normal.

## 6. Optional: data for retraining

Retraining needs the lab's hand-scored recordings, which are not in git. Copy the
`training_data` folder from the shared drive into the `sleep` folder so that you
have `sleep\training_data\Filtered Scores 2\ZZ05_BL\...`. Not needed to run the GUI.

## 7. Done. Next: run it

`RUN_THE_APP.md`. On Windows the quick way is the two scripts in
`scripts\windows\`: double-click `start_backend.bat`, then `start_gui.bat`, then
open http://localhost:3000 in Chrome or Edge.

## Windows cheat sheet for the other guides

The other guides show Mac commands. Translate like this:

| Guide says (Mac) | Type on Windows (PowerShell) |
|---|---|
| `source .venv/bin/activate` | `.\.venv\Scripts\Activate.ps1` (from the `sleep` folder) |
| `source ../.venv/bin/activate` (from `backend`) | `..\.venv\Scripts\Activate.ps1` |
| `python3` | `python` (or `py -3.11`) |
| `cd sleep/frontend` | `cd sleep\frontend` (forward slashes usually work too) |
| `export MORA_MODEL_PATH=...` or `MORA_MODEL_PATH=... uvicorn ...` | `$env:MORA_MODEL_PATH = "..."` on its own line, then the command |
| `rm backend/models/*.pkl` | `Remove-Item backend\models\*.pkl` |
| `cp a b` | `Copy-Item a b` |
| `cat file` / `sed -n '1,20p' file` | `Get-Content file` / `Get-Content file -First 20` |
| `ls` | `dir` (or `ls`, both work) |
| Ctrl+C to stop a server | same |

Paths with quotes: `"training_data\Filtered Scores 2"` works in PowerShell
exactly as the guides show with forward slashes.

## Troubleshooting

- `Activate.ps1 cannot be loaded because running scripts is disabled`: do step 2.
- `'uvicorn' is not recognized` or `'pytest' is not recognized`: the venv isn't
  active (no `(.venv)` in the prompt). Run `.\.venv\Scripts\Activate.ps1` from the
  `sleep` folder.
- `pip install` fails compiling something: your Python isn't 3.11. `python --version`
  to check; `py -3.11 -m venv .venv` to build the env with 3.11 explicitly (delete
  the old `.venv` folder first).
- `npm` errors mentioning `EPERM` or locked files: close any editor that has the
  `frontend` folder open, delete `frontend\node_modules`, and rerun `npm install`.
- Antivirus slowing `npm install` to a crawl: it is scanning thousands of small
  files; wait, or exclude the `sleep` folder.
- Long path errors (`ENAMETOOLONG`): keep the repo close to the drive root, e.g.
  `C:\code\sleep`.
