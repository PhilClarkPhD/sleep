# 2025-01-03
## TODO
- Issue: features are generated at run-time in Mora, w/ hardcoded feature selection. Leads to incompatibility between model artifacts and Mora features.
  - Solution: Change data_processing into feature store and track it w/ versioning. In Mora, reference relevant feature store version for feature selection.
- Make generating the model cards a config option
- Validations: eg validate monotonic increasing epoch if using for time series index

## 2026-11-13
### CHANGED
- **OOD is now a recording-level warning, not per-epoch blanking.** The Mahalanobis
  detector (q=0.999) flagged 15-30% of epochs on globally shifted animals (ZZ05/ZZ27/ZZ30)
  that the model still scores at ~0.9 accuracy. Backend now marks epochs `Unscored` only
  for signal-QC failures (Layer A: flat/clipped/non-finite/gross amplitude) and reports
  OOD %, QC-fail % and EMG dynamic range (p05/p50 of per-epoch EMG std) as a
  `reliability` block on the scoring response. Thresholds: QC > 1%, EMG range > 0.6,
  OOD > 10% (10-animal cohort, see docs/REM_RECALL_EXPERIMENT.md). Verified on raw WAVs:
  ZZ05 EMG range 0.91 (warn), ZZ27 QC fail 2.1% (warn), ZZ25 clean.
- New module `sleep_features/reliability.py` (`emg_dynamic_range`, `assess_recording`).
  Signal-QC `Unscored` now applies to every model (was gated on the model shipping an
  OOD detector). Epoch label is `Unscored` (lab convention / frontend type), not `Unscorable`.
- Frontend: amber reliability banner in Scoring Results; CLI: warning block after the
  summary table (both `--api` and local). CLI `--api` path fixed to read `epochs` from the
  response (was reading a nonexistent `scores` key).
- `railway.json` moved to repo root with `dockerfilePath: backend/Dockerfile` so the
  Railway build context is the repo root (needed for `sleep_features/`). Railway service
  Root Directory must be set to `/`.

## 2026-10-25
### ADDED
- 5 new hand-scored recordings (ZZ25, ZZ27, ZZ29, ZZ30, ZZ32 baseline days) under
  `training_data/Filtered Scores 2/` alongside the original 5 -> 10 animals total.
- `sleep_features.discovery.find_scores_file`: ordered label-file resolution
  (`{stem}_scores.txt` > `*scores*10s*.csv` > `*(corrected).txt` > `*(filtered).txt`
  > plain `*_scores.csv`) plus an optional `labels.json` override. Choice is logged
  and recorded in the feature-store manifest.
- Feature stores (robust_mad, 10 animals): v2 `features_2026-10-25_22-50-27.csv`,
  v1 `features_2026-10-25_22-51-28.csv` (85,687 epochs; 4,499 Unscored).
- `rem_phase23_experiment.py`: no-filter pipelines (`*-none`), per-animal breakdown,
  `--skip-hmm-nested`.

- `labels.json` value `null` now EXCLUDES a recording from discovery. ZZ01 (weak-theta recording)
  excluded via `training_data/Filtered Scores 2/labels.json` (2026-10-26). Cohort = 9 animals.
- `model/diagnose_animals.py`, `model/emg_scaling_experiment.py`, `model/local_contrast_experiment.py`
  (per-animal REM diagnosis; EMG floor-ceiling scaling = adopted for v2.1; local contrast = null result).
- `model_cards/SleepScoring_v2.1_candidate.html` — stakeholder card, 9-animal LOO.

### NOTES
- Each new animal ships two score files that disagree (kappa 0.62-0.88). The
  `(10s)` / `10sec` variants were taken as ground truth (they carry `Unscored`
  epochs and ~7-9% REM like the hand-scored originals; the plain `_scores.csv`
  files look like raw model exports). Override with `labels.json` if wrong.

## 2025-01-07
- [FIXED] Rule based filter now working
- [ADDED] test script for rule based filter

## 2025-01-03
### FIXED
- Fixed filename parsing for txt files.

## 2024-07-20
### ADDED
- `use_rule_based_filter` config option, implemented in `main.py`
- model_config loaded into make_model_card to access the above config option
- Updated sleep_functions.modify_scores(), but it is not currently working!
