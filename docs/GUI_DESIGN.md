# GUI design

What the web app does, how it is organized, and the design decisions behind it.
To change it, see `docs/guides/EDIT_THE_UI.md`.

## Design goals

1. **No installation.** Open a URL, drop a file, get scores. Everything heavy runs
   on the server.
2. **Review, not just results.** A scorer must be able to see EEG, EMG, and the
   spectrum for the current epoch in context, and fix the model's call with one key.
3. **Keyboard-first scoring.** W / E / R / T mirror the lab's existing habit; each
   key advances to the next epoch so a full pass is just a stream of keystrokes.
4. **Export that matches the lab's files.** CSV with `epoch,score`, re-importable,
   plus a light/dark breakdown.
5. **Honest about uncertainty.** Unscored epochs and recording-level warnings are
   shown, never hidden.

## Page layout (top to bottom)

```
┌───────────────────────────────────────────────────────────────────────┐
│ Mora Sleep Scoring                    Model: XGBoost v1.2.4      ⚙    │  header
├───────────────────────────────────────────────────────────────────────┤
│ [API key panel, collapsed unless the server needs a key]              │
│ [status banner: Connecting… / Authentication required / API error]    │
├───────────────────────────────────────────────────────────────────────┤
│ Upload box (drag & drop .wav)   Baseline epoch [2]                    │  FileUpload
├───────────────────────────────────────────────────────────────────────┤  ── after scoring ──
│ ◀ Prev  Epoch 1234 / 8640  Next ▶   Find  Next REM   Window [5]  Reset Y │
│ ┌──────── EEG (3 cols) ──────────┐ ┌ Power spectrum ┐                 │  EpochViewer
│ │ shaded by state, current epoch │ │ 0–50 Hz        │                 │
│ └────────────────────────────────┘ └────────────────┘                 │
│ ┌──────── EMG (3 cols) ──────────┐ ┌ delta / theta  ┐                 │
│ └────────────────────────────────┘ └────────────────┘                 │
│ Score as: [Wake] [Non REM] [REM] [Unscored]          current: REM     │
├───────────────────────────────────────────────────────────────────────┤
│ Hypnogram (full width, click to jump, light/dark shading)             │  Hypnogram
├──────────────────────┬──────────────────────┬─────────────────────────┤
│ Scoring Results      │ Recording Settings   │ Export                  │  3-column grid
│ counts, %, banner,   │ start time,          │ folder, scores CSV,     │
│ filter toggle        │ light/dark phases    │ breakdown, import, clear│
├──────────────────────┴──────────────────────┴─────────────────────────┤
│ Keyboard shortcuts help                                               │
└───────────────────────────────────────────────────────────────────────┘
```

Before a file is scored, a "How to use" list sits where the results will be.

## Components (one file each, `frontend/src/components/`)

| Component | Responsibility |
|---|---|
| `FileUpload` | Drop zone (react-dropzone), baseline epoch input, upload progress, "Score another file" |
| `EpochViewer` | The scoring workbench: navigation bar, window size, plot grid, score buttons, keyboard handler, Find dialog |
| `SignalPlot` | One Plotly trace (EEG or EMG) for `windowSize` epochs centred on the current one, background shaded by state, Y-axis lock |
| `PowerSpectrum` | Current epoch's 0–50 Hz spectrum, Y-axis lock |
| `RelativePower` | Bar chart of delta and theta relative power for the current epoch |
| `YAxisControl` | The Lock/Unlock button with min/max inputs overlaid on the plots |
| `Hypnogram` | Step plot of state vs time for the whole recording; click to navigate; light/dark shading; clock-time axis when a start time is set |
| `ScoringResults` | Counts and percentages, reliability banner, Unscored count, model version, sample rate, filter toggle |
| `RecordingSettings` | Start date/time, light/dark phase editor with overlap check, 12h presets |
| `ExportPanel` | Folder picker, Export Scores, Export Breakdown, Import Scores, Clear Scores, unsaved-changes warning |
| `ApiKeySettings` | Gear panel: enter/show/clear the API key (stored in localStorage as `mora_api_key`) |

Supporting files: `App.tsx` (layout + health check on load), `store/useAppStore.ts`
(all state), `api/sleepApi.ts` (every HTTP call), `utils/fileSave.ts` (save to
chosen folder → save dialog → download, whichever the browser supports),
`types/scoring.ts` (types mirroring the API + the colour constants).

## State (one Zustand store)

All state is in `useAppStore`. Components read what they need and call actions;
nothing is passed down through props beyond trivial values.

| Group | Fields | Actions |
|---|---|---|
| Upload | `uploadState` (idle/uploading/processing/complete/error), `uploadProgress`, `errorMessage`, `fileName`, `baselineEpoch` | `setUploadState`, `setError`, `reset` (keeps the export folder) |
| Results | `epochs` (shown), `filteredEpochs`, `unfilteredEpochs`, `showFiltered`, `summary`, `reliability`, `modelVersion`, `samplerate`, `signalData`, `modelInfo` | `setScoringResults`, `toggleFilter` |
| Navigation | `currentEpoch`, `windowSize` | `setCurrentEpoch`, `goToNext/PrevEpoch`, `goToNextREM`, `setWindowSize` |
| Editing | `editedEpochs` (Set), `hasUnsavedChanges` | `updateEpochScore` (also recomputes the summary), `clearAllScores`, `importScores` |
| Y-axes | `yAxisRanges {eeg, emg, power}` (`[min,max]` or null = auto) | `setYAxisRange`, `resetYAxisRanges` |
| Light/dark | `recordingStartTime`, `lightDarkPhases` | add/update/remove/clear phase, `getPhaseForEpoch`, `getTimestampForEpoch` |
| Export | `exportDirHandle` | `setExportDirHandle` |

Nothing is persisted. A reload clears the session; that is deliberate for now
(no server-side accounts), and the unsaved-changes warning exists because of it.

## Interaction details worth knowing

- **Keys are ignored while typing** in any text box, so Find and the phase editor
  don't trigger scoring.
- **Scoring advances.** W/E/R/T and the on-screen buttons score the current epoch
  then move to the next. This makes a review pass fast but means a mis-key needs
  ← then the right key.
- **Filter toggle re-points `epochs`** at either the filtered or unfiltered list.
  Edits are applied to whichever list is shown; switching after editing is confusing,
  so edit in one mode.
- **Y-axis lock** is per plot; dragging to zoom vertically also locks; double-click
  unlocks. Locked ranges persist across epoch navigation (that was a user request).
- **Export folder** uses the File System Access API (Chrome/Edge only). Other
  browsers fall back to a save dialog or plain download.
- **Reliability banner** is amber with one line per tripped metric, and explicitly
  says scores are still provided.

## Visual conventions

- Tailwind utility classes inline; white cards (`rounded-lg shadow`) on a
  `bg-gray-50` page, `max-w-7xl` container, 3-column grid at the `lg` breakpoint.
- State colours (`types/scoring.ts`): Wake amber `#f59e0b`, Non REM blue
  `#3b82f6`, REM green `#22c55e`, Unscored grey `#9ca3af`. Light (0.2 alpha)
  variants shade the signal plots; dark (0.5) marks the current epoch. Change them
  in one place.
- Charts are Plotly (`react-plotly.js`), configured per component.

## Known rough edges

- Uppercase `.WAV` is not stripped from the export filename (`X.WAV_scores.csv`).
- `datetime` in exports is UTC, not local time.
- The baseline-epoch input is meaningless for `robust_mad` models and should be
  removed once one is served (`docs/MODEL.md`, backlog item 3).
- No session persistence: closing the tab loses edits.
