# Data formats

Everything the pipeline reads or writes, with the rules it uses to find files.

## Recordings (WAV)

- Stereo WAV. Channel 1 (index 0) = EEG, channel 2 (index 1) = EMG.
- Sample rate is read from the file. The lab's recordings are 1000 Hz, int16,
  about 24 h (86 million samples per channel, roughly 350 MB).
- A mono file is accepted and treated as EEG with a silent EMG. Don't rely on it.
- Epochs are 10 seconds: epoch `i` is samples `i*10*samplerate` to `(i+1)*10*samplerate`.
  A trailing partial epoch is dropped.

## Training data layout (`training_data/`, not in git)

The builder walks the input folder recursively for `*.wav`. The convention is one
folder per recording:

```
training_data/Filtered Scores 2/
├── labels.json                      (optional overrides, see below)
├── ZZ05_BL/
│   ├── ZZ05_BL.wav
│   ├── ZZ05_BL.txt                  hand scores, LabChart export
│   └── ZZ05_BL(filtered).txt        hand scores, reviewed, artifacts marked Unscored
├── ZZ27_BL/
│   ├── ZZ27_BLSel.wav
│   └── ZZ27_BLSel_scores (10s).csv  hand-corrected export from the web app
└── ...
```

**Recording ID.** The WAV stem is split at its last underscore: `ZZ05_BL.wav` →
ID `ZZ05`, day `BL`, `ID_day = "ZZ05_BL"`. `ID_day` is the grouping key everywhere
(train/test split, leave-one-animal-out, manifest). Duplicate `ID_day`s are dropped
with a warning (first in sorted path order wins).

**Which label file is used.** For each WAV, the first match in its own folder
(case-insensitive):

1. `labels.json` entry (see below)
2. `<stem>_scores.txt`
3. `*scores*10s*.csv`, then `*10sec*scores*.csv` (hand-corrected web-app exports)
4. `*(corrected).txt`, then `*(filtered).txt`
5. `*_scores.csv` (warns: this pattern is usually raw model output, not human labels)

WAVs with no match are skipped with a warning. The choice is logged and written to
the feature-store manifest.

**`labels.json`** in the walk root or a recording folder (folder-level wins):

```json
{
  "_comment": "ZZ01: weak-theta electrode, excluded 2026-10-26",
  "ZZ01_BLSel24hr": null,
  "ZZ27_BLSel": "ZZ27_BLSel_scores (10s).csv"
}
```

Keys are WAV stems. A filename value forces that label file. `null` excludes the
recording from the build entirely. Keys starting with `_` are ignored.

**Baseline epoch file** `<stem>_epoch.txt` (one integer): only needed for the
legacy `baseline_epoch` normalization. Not used by `robust_mad`.

## Label (score) files

Comma-separated; `.txt` or `.csv` doesn't matter. Two layouts are recognized by
their header:

| Layout | Header | Epoch numbering |
|---|---|---|
| simple (web app export, our CSVs) | `epoch,score[,timestamp_seconds,datetime,phase]` | from 0 |
| LabChart | `Epoch #,Start Time,End Time,Score #, Score` | from 1 (converted) |

Column names are matched after lowercasing and stripping spaces; `epoch`, `epoch #`,
`epoch#` all work. A file with only a `score` column is read in row order.
Malformed lines and blank labels are dropped.

Labels: `Wake`, `Non REM`, `REM`. Anything else, or a missing epoch, becomes
`Unscored`. Training drops `Unscored` rows; the app uses `Unscored` for epochs that
fail signal QC and for manual "T" marks. When importing, `nrem`, `non-rem`, `wake`,
`rem` are normalized.

Lab conventions that matter for the model (settled 2026-10-26, don't re-derive):

- Labels are **unfiltered**: single-epoch REM bouts count as real REM. The older
  cohort's files have no single-epoch bouts because of how they were scored, not
  because of a filter.
- The `(filtered)` suffix on old files means "reviewed, artifacts marked Unscored",
  not "smoothed". Prefer `(filtered)`/`(corrected)` over the plain file.
- For the newer cohort, the `(10s)` CSVs are the human-corrected labels; the plain
  `_scores.csv` next to them are old model output. Never train on the latter.

## Feature store (`feature_store/`, not in git)

`python -m sleep_features build` writes a pair:

- `features_<YYYY-MM-DD_HH-MM-SS>.csv`: columns `ID, day, ID_day, epoch`, then the
  feature set's columns, then `score`. One row per epoch per recording.
- `features_<timestamp>.manifest.json`: `feature_code_version`, `feature_set`,
  `normalization {strategy, emg_event_k}`, `git_sha`, `config` (bands, epoch
  length, column lists), `totals` (rows, label distribution), and `recordings`
  (per recording: id, wav path, label file used, epochs, scored epochs).

`train.py` copies the manifest's normalization and feature-code version into the
model's metadata; that is how the backend knows how to featurize for that model.
A CSV without a manifest (pre-2026 builds) produces a model that the backend will
serve with the legacy baseline-epoch path.

## Model artifact (`model_artifacts/<model_id>/`, committed)

- `<model_id>.pkl`: joblib tuple `(XGBClassifier, metadata: dict, LabelEncoder)`.
  Metadata keys the serving code reads: `model_version`, `model_name`, `notes`,
  `feature_cols`, `normalization.strategy`, `feature_code_version`,
  `ood_detector.file`. Also stored: `best_params`, `search_space`, holdout
  `test_*` metrics, `feature_importances`, `feature_store_table`,
  `feature_store_manifest`, timestamps.
- `<model_id>_ood.pkl`: `sleep_features.ood.MahalanobisOOD`, fit on the training
  features. Must sit next to the model `.pkl` wherever it is served.
- `<model_id>_test_data.csv`: holdout features + `score` + `predicted_score`.

## API responses

`POST /api/v1/score` returns JSON (full schema at http://localhost:8000/docs):

- `epochs` / `unfiltered_epochs`: list of `{epoch, score, timestamp_seconds}`.
  `epochs` has the rule-based filter applied; the GUI shows `unfiltered_epochs`
  by default.
- `summary`: counts and percentages per state (percentages over scorable epochs),
  duration, `unscorable_count`.
- `reliability`: `{level, qc_fail_percent, emg_range_p05_p50, ood_percent, warnings[]}`.
- `signal_data` (if requested): per-epoch downsampled EEG/EMG (500 points), power
  spectrum 0–50 Hz at 0.1 Hz, delta and theta relative power.

## GUI exports

- `<name>_scores.csv`: `epoch,score,timestamp_seconds` plus `datetime`
  (`YYYY-MM-DD HH:MM:SS`, if a start time was set) and `phase` (`light`/`dark`/blank,
  if phases were set). Re-importable.
- `<name>_breakdown.csv`: `Stage,Epochs,Proportion` for Wake / Non REM / REM,
  then the same per light and dark phase when configured. Proportions exclude
  Unscored epochs.
