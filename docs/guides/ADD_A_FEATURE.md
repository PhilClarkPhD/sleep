# Guide: test a new feature

"Feature" here means a per-epoch number the model gets as input (e.g. relative
theta power), not a GUI feature. For GUI changes see `EDIT_THE_UI.md`.

Read `RETRAIN_MODEL.md` first; this guide adds one step before it and one rule.

## The rule: one code path

Features are computed by `sleep_features/` both when building the training table and
when the backend scores a new file. **Never compute a feature in a notebook and
paste the column into the CSV.** If the model learned from it, the backend must be
able to compute it identically. The `FEATURE_CODE_VERSION` constant exists to
catch mismatches.

## Where features live

| File | What's there | Add here if your feature is... |
|---|---|---|
| `sleep_features/transforms.py` | FFT power, band power, time-domain amplitude stats (`compute_signal_features_raw`), Hjorth | a time-domain or band-power statistic |
| `sleep_features/spectral.py` | sigma/beta/gamma relative power, spectral entropy, spectral edge | a spectral shape statistic |
| `sleep_features/temporal.py` | `_roll3` (centered 3-epoch mean), `_diff1` (change from previous epoch) | context from neighbouring epochs |
| `sleep_features/normalize.py` | which columns get per-recording scaling | an amplitude feature (it must be normalized; ratios must not) |
| `sleep_features/config.py` | the column lists: `GENERATED_FEATURE_COLS` (v1), v2 extras, `MODEL_FEATURE_COLS` | always: register the column name |
| `sleep_features/pipeline.py` | `generate_feature_table` wires the above together | only if you add a new module |

Existing features, for orientation (v1): `EEG_quantile_80`, `EEG_ptp`, `EEG_ss`,
`EMG_std`, `EMG_events`, `EMG_ptp`, `delta_rel`, `theta_rel`, `theta_over_delta`
(the 9 the model uses), plus `EEG_quantile_diff`, `EEG_amp`, `EEG_std` (computed
but unused). v2 adds `sigma_rel`, `beta_rel`, `gamma_rel`, `spectral_entropy`,
`spectral_edge_95`, three Hjorth parameters, and `_roll3`/`_diff1` versions of four
columns.

## Step-by-step

1. **Implement** the column in the appropriate module. It must be a function of the
   epoch's raw EEG/EMG (and, for temporal features, neighbouring epochs' values)
   with no dependence on labels.
2. **Register** it in `sleep_features/config.py`. Add it to the v2 list (or a new
   `v3` set) so v1 stays unchanged.
3. **Decide normalization.** Amplitude-like (units of microvolts) → add to the list
   in `normalize.py` so it gets per-recording median/MAD scaling. Ratio or
   dimensionless → leave raw.
4. **Bump `FEATURE_CODE_VERSION`** in `config.py`: MINOR (`1.0.0` → `1.1.0`) for a
   new column with old columns unchanged; MAJOR if you changed how an existing
   column is computed (every existing model would then be refused by the backend
   until retrained, which is the point).
5. **Test it.** Add a unit test in `tests/unit/test_feature_set_v2.py` (or a new
   file) that builds a tiny synthetic recording and checks the column exists, is
   finite, and responds as expected (e.g. higher for a synthetic theta burst). Run:
   ```bash
   pytest tests/unit -q
   ```
   `tests/golden/generate_features_sample.csv` pins the v1 columns' values; if that
   test fails you changed existing math (MAJOR bump, deliberate or not).
6. **Build a store** with your feature set:
   ```bash
   python -m sleep_features build --input "training_data/Filtered Scores 2" \
       --out feature_store --feature-set v2 --normalization robust_mad
   ```
7. **Compare with leave-one-animal-out**, which is the honest test for "does this
   feature help on a new rat":
   ```bash
   python -m model.experiments.compare_feature_sets \
       --v1 feature_store/features_<baseline build>.csv \
       --v2 feature_store/features_<your build>.csv
   ```
   Look at pooled kappa and REM recall, and the per-animal table. A feature that
   helps the average but hurts two animals is suspicious.
8. **If it wins**, add the column to `feature_cols` in `model/model_config.json`,
   retrain per `RETRAIN_MODEL.md` with a MINOR version bump, promote.

## What has already been tried (don't repeat)

Documented in `docs/REM_RECALL_EXPERIMENT.md`:

- v2 spectral + Hjorth + temporal features: small gain, adopted for the v2.1 candidate.
- Class weights / threshold tuning for REM: modest gain, weights 1–2 only.
- HMM / Viterbi smoothing: raises REM recall but kills single-epoch REM, which the
  lab counts as real. Ruled out.
- Local-contrast theta features for short REM bouts: no gain.
- Floor-ceiling EMG scaling (per-recording p05–p95): helps the low-EMG-gain animal;
  recommended for v2.1 but **not yet implemented in `normalize.py`**. This is the
  most valuable open item.

## The open item in detail: floor-ceiling EMG scaling

Replace median/MAD with `(x - p05) / (p95 - p05)` for `EMG_std` and `EMG_ptp` only.
Prototype in `model/experiments/emg_scaling_experiment.py` (`apply_scaling`).
To productionize: add a strategy in `normalize.py` (e.g. `robust_mad_emgfc`),
expose it in the build CLI choices, bump `FEATURE_CODE_VERSION` MINOR, make the
backend's scoring service accept the new strategy name, rebuild, retrain.
