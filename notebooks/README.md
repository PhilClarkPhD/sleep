# Notebooks (legacy)

Exploration notebooks from 2024–2025. They have hardcoded paths to folders that no
longer exist (`sleep_data/`, old feature stores) and predate the `sleep_features`
package. Kept for the record; nothing depends on them.

| Notebook | What it was |
|---|---|
| `FS-peak_to_peak.ipynb` | Exploring the EEG peak-to-peak feature on one WAV |
| `model_card.ipynb` | Evidently model card for an old model (see `model/experiments/legacy_make_model_card.py`) |
| `modify_scores.ipynb` | Hand-editing the `XGBoost_1.2.3-test` holdout file |
| `train_v_test_score_imbalance_EDA.ipynb` | Class-imbalance EDA on a 2024 feature store |

For current analysis, use the scripts in `model/experiments/` (documented there) or
start a fresh notebook that imports `sleep_features` and `model.diagnostics`.
