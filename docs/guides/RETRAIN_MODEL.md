# Guide: retrain the model

When: you have new hand-scored recordings, corrected labels, or want to include or
exclude an animal. Budget about an hour of wall-clock time, most of it waiting.

Prerequisites: `SETUP.md` done, `training_data/` in place (`docs/DATA_FORMATS.md`).
All commands run from the repo root with the venv active:

```bash
cd sleep && source .venv/bin/activate
```

The pipeline has three steps. Each produces a file the next one reads.

```
training_data/   ──(1) build feature store──▶  feature_store/features_<timestamp>.csv (+ .manifest.json)
feature store    ──(2) train────────────────▶  model_artifacts/<model_id>/<model_id>.pkl, _ood.pkl, _test_data.csv
model artifact   ──(3) promote──────────────▶  backend/models/  (then restart the backend)
```

## Step 1: build the feature store

This turns every WAV + label file under `training_data/` into one big CSV of
per-epoch features. About 2 minutes per 24-hour recording.

```bash
python -m sleep_features build \
    --input "training_data/Filtered Scores 2" \
    --out feature_store \
    --feature-set v1 \
    --normalization robust_mad
```

What the options mean:

- `--input`: folder searched recursively for `*.wav`. Each WAV needs a label file
  in the same folder (rules in `docs/DATA_FORMATS.md`). Recordings without labels are
  skipped with a warning. A `labels.json` in that folder can override which label
  file is used, or exclude a recording with `null`.
- `--feature-set v1`: the 12 original columns, of which the model uses 9. `v2` adds
  extended spectral, Hjorth, and temporal-context columns (see `ADD_A_FEATURE.md`).
  **Use `v1` unless you are deliberately changing the model's inputs**, because the
  current `model_config.json` lists the 9 v1 columns.
- `--normalization robust_mad`: per-recording median/MAD scaling. Always use this.
  `baseline_epoch` is the legacy method and needs `*_epoch.txt` files.
- `--limit N`: build only the first N recordings (quick test).
- `--emg-event-k`: leave at default. It is recorded in the manifest but not currently
  applied by the pipeline.

Success looks like: a log line per recording (`ZZ05_BL: 8640 epochs, 8601 scored,
labels from ZZ05_BL(filtered).txt`), then two new files in `feature_store/`:
`features_<timestamp>.csv` and `features_<timestamp>.manifest.json`.

**Read the log.** Check that every animal you expect is listed, that the chosen
label file is the one you intended, and that nothing was excluded by surprise.
The manifest's `recordings` list records the same information permanently.

## Step 2: train

Edit `model/model_config.json`:

1. `feature_store_table`: the CSV name from step 1 (file name only; the directory is
   `feature_store/`).
2. `model_version`: bump it. The convention is `MAJOR.MINOR.PATCH`, e.g. `1.2.4` →
   `1.2.5` for new data with the same features, `1.3.0` for a changed feature set.
   The output folder is `model_artifacts/XGBoost_<version>/`, and **re-using a
   version overwrites that folder**.
3. `notes`: one line on what changed. It is stored in the model and shown in the API.

Leave the rest alone unless you know why. `feature_cols` must be columns that exist
in the feature store.

Then:

```bash
python -m model.train --config model/model_config.json
```

What happens, in order (watch the log):

1. Loads the CSV and its manifest. Drops `Unscored` epochs.
2. Splits each recording in time: first 75% of its epochs → train, last 25% → test.
   (Not a random split: neighbouring epochs are near-duplicates, so random splits
   give inflated scores.)
3. Hyperparameter search: 50 random combinations × 5-fold time-series CV with a
   6-epoch gap. **This is the slow part, 10 to 40 minutes.** `--skip-tuning` reuses
   a `best_params` block if the config has one.
4. Holdout evaluation on the 25% test slice: prints kappa, balanced accuracy,
   per-class precision/recall/F1, transition accuracy. Note these numbers.
5. Refits on 100% of the data, fits the OOD detector, saves.

Output in `model_artifacts/XGBoost_<version>/`:

| File | What |
|---|---|
| `XGBoost_<version>.pkl` | `(model, metadata, label_encoder)`. Metadata includes feature columns, normalization strategy, feature-code version, holdout metrics, best params, feature importances |
| `XGBoost_<version>_ood.pkl` | Out-of-distribution detector, used for reliability warnings |
| `XGBoost_<version>_test_data.csv` | The holdout slice with predictions, for `evaluate.py` |

## Step 3: decide whether it's better

Compare the holdout numbers to the previous model (its metadata has them too):

```bash
python -m model.evaluate --model model_artifacts/XGBoost_1.2.4/XGBoost_1.2.4.pkl
python -m model.evaluate --model model_artifacts/XGBoost_<new>/XGBoost_<new>.pkl
```

Caution: each model's test slice is different if the data changed, so this is a
rough comparison. For a fair one, evaluate both on the same labeled CSV with
`--data`. The honest cohort-level estimate is leave-one-animal-out, which
`model/experiments/compare_feature_sets.py` does. See `EVALUATE_MODEL.md`.

Rules of thumb: kappa within 0.01 of the old model and REM recall not lower is
"same". A new animal that scores badly on its own is more likely a recording or
label problem than a model problem; run `python -m model.experiments.diagnose_animals --help`.

## Step 4: promote to the app

```bash
rm backend/models/*.pkl
cp model_artifacts/XGBoost_<new>/XGBoost_<new>.pkl backend/models/
cp model_artifacts/XGBoost_<new>/XGBoost_<new>_ood.pkl backend/models/
```

The backend serves whatever single `.pkl` is in `backend/models/` (the newest if
there are several). Restart it (`RUN_THE_APP.md`) and confirm
http://localhost:8000/api/v1/model/info shows the new version. Score
`sample_data/sample_signal.wav` and a real recording in the GUI as a smoke test.

Commit and push so the repo records which model is current. (If the app is ever
hosted again, the Docker image copies `backend/models/` in, so the committed `.pkl`
is what gets served; see `DEPLOY.md`.)

## Step 5: record it

Add an entry to `model/CHANGELOG.md`: date, version, what data, holdout kappa and
REM recall, and anything you decided (e.g. "ZZ27 excluded, labels disputed").
Future-you will want it.

## Common problems

- `KeyError: 'EMG_std'` or similar at training: `feature_cols` in the config names a
  column that isn't in the store. Check the store's header and `--feature-set`.
- `baseline_epoch normalization needs a *_epoch.txt`: you passed the legacy
  normalization. Use `robust_mad`.
- The new model is served with a "baseline epoch" input in the GUI: the model's
  metadata has no `normalization.strategy`, meaning the feature store had no
  manifest. Rebuild the store with the current code.
- Backend refuses to start with "Incompatible feature code": the model was trained
  against a different MAJOR `FEATURE_CODE_VERSION` than the serving code. Retrain,
  or check out the matching code version.
