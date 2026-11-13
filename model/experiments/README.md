# Model experiments (research scripts)

These are research scripts, not part of the production pipeline. They were used
in October 2026 to decide the v2 feature set, the normalization strategy, and the
"no rule filter at serving" decision. Results and conclusions are written up in
`docs/REM_RECALL_EXPERIMENT.md`. Nothing in `backend/`, `cli/`, or `model/train.py`
imports from here.

All take a feature-store CSV (see `docs/guides/RETRAIN_MODEL.md` for how to build one)
and run leave-one-animal-out evaluation. Run from the repo root as modules, e.g.
`python -m model.experiments.diagnose_animals --help`.

| Script | Question it answers | Reusable? |
|---|---|---|
| `compare_feature_sets.py` | Is feature set v2 better than v1 on the same recordings? | Yes: run it whenever you add a feature |
| `diagnose_animals.py` | Which animals score badly, and why (EMG, theta, labels)? Per-animal LOO breakdown + label-free recording metrics | Yes: run it after adding a new animal |
| `rem_recall_experiment.py` | Phase 1: class weights / thresholds to raise REM recall. Defines shared constants (`GROUP_COL`, `TARGET_COL`, `_fit`) the other scripts import | Mostly one-off |
| `rem_phase23_experiment.py` | Phase 2/3: extra REM features, HMM smoothing, nested cost-sensitive tuning; defines `augment_features` | Mostly one-off |
| `emg_scaling_experiment.py` | Does floor-ceiling EMG scaling fix the low-EMG-gain animal (ZZ05)? (Answer: partly, adopted for v2.1) | One-off |
| `local_contrast_experiment.py` | Do local-contrast theta features help single-epoch REM? (Answer: no) | One-off |

Dependency order (imports): `rem_recall_experiment` <- `rem_phase23_experiment` <- `emg_scaling_experiment`,
`diagnose_animals` <- `local_contrast_experiment`. Keep them together.
