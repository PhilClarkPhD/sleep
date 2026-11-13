# Guide: open the project in PyCharm

Works with PyCharm Community (free) or Professional. Do `SETUP_WINDOWS.md` or
`SETUP.md` first so the `.venv` folder exists; PyCharm then just needs pointing at it.

## 1. Open the folder

File → Open → choose the `sleep` folder (the one containing `backend`, `frontend`,
`model`, `docs`). Click "Trust Project" if asked.

## 2. Pick the Python interpreter (the one setting that matters)

PyCharm usually detects `.venv` and offers it in a banner at the top: click
"Configure" / "OK". If not:

- Windows: File → Settings → Project: sleep → Python Interpreter → "Add
  Interpreter" → "Add Local Interpreter" → Virtualenv Environment → **Existing** →
  browse to `sleep\.venv\Scripts\python.exe`.
- Mac: PyCharm → Settings → same path → `sleep/.venv/bin/python`.

Success looks like: the interpreter dropdown at the bottom right says
`Python 3.11 (sleep)` and `import xgboost` in any file is not underlined in red.

Do **not** let PyCharm create a new environment; reuse the one setup made.

## 3. Tell PyCharm where the code roots are

Right-click the `backend` folder in the project tree → "Mark Directory as" →
**Sources Root**. The backend imports as `from app....`, and this makes those
imports resolve. The repo root is already a content root, so `sleep_features`,
`model`, and `cli` resolve on their own.

Optional but recommended, to keep indexing fast: right-click each of these →
"Mark Directory as" → **Excluded**: `frontend/node_modules`, `.venv`,
`training_data`, `feature_store`, `model_artifacts`, `model_cards`.

## 4. Test runner

File → Settings → Tools → Python Integrated Tools → "Default test runner" →
**pytest**. Then right-click `tests` → "Run 'pytest in tests'". Expect all green.

## 5. Run configurations

Two are shipped in the repo (`.run/` folder) and appear in the run dropdown at the
top right once the interpreter is set:

| Name | What it does |
|---|---|
| **Backend (uvicorn)** | Starts the API on http://localhost:8000 with auto-reload. Equivalent to `start_backend` script |
| **Tests (pytest)** | Runs the whole test suite |

If they show a red "module not found" error, PyCharm named your project module
differently: open Run → Edit Configurations, pick the configuration, and choose
your project in the "Project"/"Module" dropdown. Or recreate it: "+", Python,
switch "script" to **module**, module name `uvicorn`, parameters
`app.main:app --reload --port 8000`, working directory `backend`.

The GUI (`npm run dev`) has no Python run configuration. Start it from the
**Terminal** tab at the bottom of PyCharm (`cd frontend`, then `npm run dev`), or
double-click `scripts\windows\start_gui.bat`. PyCharm Professional users can
add an "npm" run configuration pointing at `frontend/package.json`, script `dev`.

## 6. Editing TypeScript (the GUI)

PyCharm Community highlights TypeScript but does not type-check it. Run
`npm run build` in the Terminal before committing a UI change; it fails on type
errors. PyCharm Professional (or VS Code) gives full TypeScript support if she
ends up doing a lot of UI work.

## 7. Git inside PyCharm

The Git menu / "Commit" tab (Alt+0) works out of the box once Git for Windows is
installed. Branch name is shown bottom right; it should say `claude/refactor`
(or `main` after the merge). "Update Project" = `git pull`. Commit and Push are
in the Git menu.

## Things to ignore

- `.idea/` is PyCharm's own folder and is git-ignored. Yours is created fresh.
- `.envrc`, `.tool-versions`, `.direnv/` are from the original developer's Mac
  setup (asdf + direnv). PyCharm doesn't use them.
- A banner offering to install "requirements" from `backend/requirements.txt`:
  harmless, everything is already installed via the root `requirements.txt`.
