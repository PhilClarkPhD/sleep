# Guide: evaluate a model

When: after retraining, when a new animal looks wrong, or when comparing two models.
Run from the repo root with the venv active.

## The metrics, and what "good" is

Every evaluation in this project reports the same set (`model/diagnostics.py`,
`SleepScoringMetrics`):

| Metric | Meaning | Current model |
|---|---|---|
| Cohen's kappa | Agreement with the human scorer beyond chance. The headline number. 0.6–0.8 substantial, >0.8 almost perfect | 0.84 |
| Balanced accuracy | Mean per-class recall, so REM (about 8% of epochs) counts as much as Wake | 0.88 |
| Per-class precision / recall / F1 | Recall = of the human's REM epochs, how many did we call REM. Precision = of our REM calls, how many were right | REM recall 0.80, precision 0.90 |
| Transition accuracy | Accuracy restricted to epochs where the state changes. Always lower; errors cluster at boundaries | 0.71–0.74 |
| Confusion matrix | Rows = truth, columns = prediction | REM↔NREM is the main confusion |

Plain accuracy is also printed but is misleading: predicting "never REM" scores
about 92% accuracy.

**REM recall is the number to watch.** It is the weakest and the one the lab cares
about. Interpretation of per-animal REM recall below 0.7: see "Diagnosing a bad
animal".

## Evaluate one saved model

```bash
python -m model.evaluate --model model_artifacts/XGBoost_1.2.4/XGBoost_1.2.4.pkl
```

By default this uses the model's own holdout slice (`<model_id>_test_data.csv`,
written at training time). It prints the metrics, the confusion matrix, and the
feature importances. Options:

- `--data path/to/features.csv`: evaluate on any labeled feature CSV instead
  (needs the model's feature columns plus a `score` column; `Unscored` rows are
  dropped). Use the same CSV for two models to compare them fairly.
- `--no-filter`: skip the rule-based smoothing filter. **Use this**; the app's
  default view is unfiltered and the lab counts single-epoch REM as real.
- `--output-json metrics.json`: save the numbers.

## Compare two models fairly

Same data, same settings:

```bash
python -m model.evaluate --model model_artifacts/XGBoost_1.2.4/XGBoost_1.2.4.pkl \
    --data feature_store/features_<ts>.csv --no-filter --output-json old.json
python -m model.evaluate --model model_artifacts/XGBoost_1.2.5/XGBoost_1.2.5.pkl \
    --data feature_store/features_<ts>.csv --no-filter --output-json new.json
```

Caveat: if either model was trained on part of that CSV, its numbers are optimistic.
Both trained on it → both optimistic → the comparison is still roughly fair, the
absolute values are not.

## The honest number: leave-one-animal-out (LOO)

Train on all animals but one, test on the held-out animal, repeat for every animal.
This is what the numbers in `docs/MODEL.md` and `docs/REM_RECALL_EXPERIMENT.md` are,
and it is the right estimate of "how will this do on the next rat". Slower (one
model fit per animal).

```bash
python -m model.experiments.compare_feature_sets \
    --v1 feature_store/features_<v1 build>.csv \
    --v2 feature_store/features_<v2 build>.csv
```

It prints pooled and per-animal kappa / REM recall for both feature sets. Pass the
same CSV twice to just get LOO numbers for one store.

## Diagnosing a bad animal

```bash
python -m model.experiments.diagnose_animals \
    --v2 feature_store/features_<v2 build>.csv \
    --data "training_data/Filtered Scores 2" \
    --focus ZZ05 ZZ27
```

For each focus animal it reports: where its REM errors go (NREM vs Wake), which
features separate REM worst, how far its feature distribution sits from the other
animals, raw-signal QC rate, EMG dynamic range, and (if two label files exist) how
much the labels disagree. Known patterns from the 2026-10 cohort:

| Pattern | Signature | Action |
|---|---|---|
| EMG channel with low gain (ZZ05) | EMG range p05/p50 > 0.6, REM EMG not lower than NREM | Recording problem. Model can't fix it; reliability banner warns the user |
| Weak theta electrode (ZZ01) | REM theta/delta much lower than cohort; not detectable without labels | Excluded from training via `labels.json` |
| Fragmented or disputed labels (ZZ27) | Many single-epoch REM bouts; two label files disagree on most REM | Re-review labels with the lab before trusting the number |
| Noisy recording (ZZ27, ZZ30) | QC fail > 1%, OOD > 10% | Expect lower agreement; banner warns |

## Reading the model's own record

Every `.pkl` carries its training-time metrics:

```bash
python -c "import joblib; m=joblib.load('model_artifacts/XGBoost_1.2.4/XGBoost_1.2.4.pkl')[1]; \
print({k:m[k] for k in m if k.startswith('test_') or k in ('model_version','feature_store_table','normalization','feature_code_version')})"
```

## Reference

`docs/MODEL_VALIDATION.md` explains the cross-validation design (why time-series
splits with a gap, why group splits) and the diagnostics API in more depth.
