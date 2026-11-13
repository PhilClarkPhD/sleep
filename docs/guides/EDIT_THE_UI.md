# Guide: change something in the GUI

Prerequisites: `SETUP.md` (Node installed, `npm install` done). Read
`docs/GUI_DESIGN.md` first for the component map; this guide is the mechanics.

If you are an AI assistant: the frontend is React 19 + TypeScript + Vite + Tailwind
v4 + Zustand + react-plotly.js. There is one page, one store, one API module. Keep
changes inside the component that owns the behavior; put new state in the store;
put new HTTP calls in `src/api/sleepApi.ts`. Run `npm run build` before declaring
done; it type-checks.

## The edit loop

```bash
# terminal 1: backend (needed for anything past the upload box)
cd backend && source ../.venv/bin/activate && uvicorn app.main:app --reload --port 8000
# terminal 2: frontend with hot reload
cd frontend && npm run dev        # http://localhost:3000
```

(Or the start scripts in `scripts/windows/` and `scripts/mac/`. Hot reload works
either way.)

Edit a file under `frontend/src/`, save, the browser updates in place (state is
usually kept, so you don't have to re-upload after every edit). Open the browser's
developer console (F12) to see errors.

Before committing:

```bash
cd frontend
npm run lint      # style / obvious mistakes
npm run build     # type-check + production build; must end with "built in …"
```

## Where things are

```
frontend/src/
├── App.tsx                 page layout, health check, help text, keyboard-shortcut list
├── main.tsx                entry point (don't touch)
├── index.css               Tailwind import + 3 custom classes
├── components/             one file per visible block (see GUI_DESIGN.md table)
├── store/useAppStore.ts    ALL state and actions
├── api/sleepApi.ts         ALL calls to the backend (health, model info, score)
├── utils/fileSave.ts       saving files: chosen folder → save dialog → download
└── types/scoring.ts        types mirroring the API JSON + state colours
```

## Recipes

**Change a colour, label, or threshold text.** Colours: `types/scoring.ts`
(`SLEEP_COLORS`). Help text and shortcut list: `App.tsx`. Banner wording:
`components/ScoringResults.tsx`.

**Add a keyboard shortcut.** `components/EpochViewer.tsx`, the `useEffect` with the
`keydown` listener. Keys are already ignored while an input is focused; keep that.
Add the new key to the shortcut list in `App.tsx` too.

**Add a column to the exported CSV.** `components/ExportPanel.tsx`, the function
that builds the scores CSV. If the value comes from the API, add it to the type in
`types/scoring.ts` and store it via `setScoringResults` in `useAppStore.ts`.

**Show a new value from the API.** Three places, in order: (1) `types/scoring.ts`,
add the field to `ScoringResponse` (or `ScoringStats`, `RecordingReliability`);
(2) `store/useAppStore.ts`, add a state field and set it in `setScoringResults`;
(3) the component that displays it reads it with `useAppStore()`.
`components/FileUpload.tsx` is where the response is handed to the store.

**Add a new panel.** Create `components/MyPanel.tsx` exporting a function
component, read state with `const { x } = useAppStore()`, and place it in the grid
in `App.tsx`. Copy the card wrapper classes from a neighbour
(`bg-white rounded-lg shadow p-6`).

**Change plot behaviour.** `SignalPlot.tsx` (EEG/EMG traces, shading, window),
`PowerSpectrum.tsx`, `RelativePower.tsx`, `Hypnogram.tsx`. All use the `Plot`
component from `react-plotly.js` with a `data` array and a `layout` object; the
Plotly docs apply directly. Y-axis locking is shared logic: see how `SignalPlot`
uses `yAxisRanges` and `YAxisControl`.

**Remove the baseline epoch input** (planned once a `robust_mad` model is served):
delete the input in `FileUpload.tsx`, the `baselineEpoch` field/action in the
store, and stop sending `start_epoch` in `sleepApi.scoreFile`. The API will keep
accepting the field, so nothing breaks if you do it in stages.

**Talk to a new backend endpoint.** Add a function in `api/sleepApi.ts` next to
`getModelInfo`; it already handles the base URL and the `X-API-Key` header.

## Conventions

- TypeScript strict: no `any` unless you're wrapping Plotly types.
- Tailwind classes inline; no CSS modules. If a class combination repeats three
  times, consider a small component, not a stylesheet.
- State updates go through store actions; components never mutate arrays in place.
- Keep `types/scoring.ts` in sync with `backend/app/schemas/scoring.py`. The
  backend's `/docs` page shows the live schema.

## Testing a change

There is no frontend test suite. Test by hand: upload `sample_data/sample_signal.wav`
(90 min, all Wake, fast) and a real 24-hour file (exercises window sizes, Next REM,
export). Check Chrome and one of Safari/Firefox if you touched export or file
handling.

## Sharing the change

Commit and push to `claude/refactor`. Anyone running locally gets it with
`git pull` and a restart of `npm run dev`. Hosting is optional; see `DEPLOY.md`.
