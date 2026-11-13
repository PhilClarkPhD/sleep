# Model Validation Guide

This document explains the validation approach for the Mora sleep scoring model, including proper cross-validation for time-series data and comprehensive evaluation metrics.

---

## Why Special Validation for Sleep Scoring?

Standard machine learning validation (random k-fold CV) is problematic for sleep data because:

1. **Temporal autocorrelation**: Adjacent epochs are highly correlated - a rat in NREM at epoch 100 is likely still in NREM at epoch 101.

2. **Data leakage**: Random shuffling can put epoch 101 in training while epoch 100 is in test, allowing the model to "cheat" by learning from nearly identical data.

3. **State persistence**: Sleep states persist for minutes, so random splits overestimate true generalization.

---

## Cross-Validation Strategies

### 1. Group Time-Series Split (Recommended)

```python
from model.cross_validation import get_cv_splitter

cv = get_cv_splitter(
    strategy="group_time_series",
    n_splits=5,
    gap=6,  # 1 minute gap to prevent autocorrelation leakage
)

# Use in hyperparameter search
best_params, duration = find_best_params(
    X_train, y_train, search_space,
    cv_splitter=cv,
    groups=train_df['ID_day']
)
```

**How it works:**
- Respects temporal ordering within each recording (ID_day)
- Uses expanding window: later folds have more training data
- Optional gap between train/test prevents adjacent-epoch leakage

```
Recording 1:  [====TRAIN====]--gap--[TEST][..........]
Recording 2:  [====TRAIN====]--gap--[TEST][..........]
Recording 3:  [====TRAIN====]--gap--[TEST][..........]
                   Fold 1

Recording 1:  [=======TRAIN=======]--gap--[TEST][...]
Recording 2:  [=======TRAIN=======]--gap--[TEST][...]
Recording 3:  [=======TRAIN=======]--gap--[TEST][...]
                   Fold 2
```

### 2. Leave-One-Recording-Out

```python
cv = get_cv_splitter(strategy="leave_one_out")
```

**How it works:**
- Each fold trains on all recordings except one
- Tests on the held-out recording
- Strictest test of generalization to new subjects

**Tradeoffs:**
- Higher variance (depends heavily on which recording is held out)
- Better estimate of true generalization
- Fewer effective training samples per fold

---

## Evaluation Metrics

### Standard Metrics

| Metric | Use Case | Notes |
|--------|----------|-------|
| **Cohen's Kappa** | Inter-rater agreement | Standard in sleep literature. Adjusts for chance agreement. |
| **Balanced Accuracy** | Class imbalance | Macro-average of recall. Better than accuracy when REM is rare. |
| **F1 Weighted** | Overall performance | Weights by class support. |
| **F1 Macro** | Equal class importance | Unweighted average. Penalizes poor REM detection. |

### Interpreting Cohen's Kappa

| Kappa | Interpretation |
|-------|----------------|
| < 0.20 | Slight agreement |
| 0.21 - 0.40 | Fair |
| 0.41 - 0.60 | Moderate |
| 0.61 - 0.80 | Substantial |
| 0.81 - 1.00 | Almost perfect |

For sleep scoring, aim for **κ > 0.80** to match expert human agreement.

### Using the Diagnostics Module

```python
from model.diagnostics import evaluate_model, print_confusion_matrix

# Get comprehensive metrics
metrics = evaluate_model(y_test, y_pred)

# Print human-readable summary
print(metrics)

# Access individual metrics
print(f"Cohen's Kappa: {metrics.cohen_kappa:.3f}")
print(f"REM Recall: {metrics.class_metrics['REM']['recall']:.3f}")

# View confusion matrix
print(print_confusion_matrix(y_test, y_pred))
```

**Example output:**
```
============================================================
SLEEP SCORING MODEL EVALUATION
============================================================

OVERALL METRICS:
  Accuracy:          0.8934
  Balanced Accuracy: 0.8567
  Cohen's Kappa:     0.8234  (Substantial)
  F1 (weighted):     0.8912
  F1 (macro):        0.8456

PER-CLASS METRICS:
  Non REM:
    Precision: 0.9123
    Recall:    0.9456
    F1:        0.9287
    Support:   4521
  REM:
    Precision: 0.8234
    Recall:    0.7891
    F1:        0.8059
    Support:   892
  Wake:
    Precision: 0.8912
    Recall:    0.8354
    F1:        0.8624
    Support:   2103

TRANSITION ANALYSIS:
  Accuracy at transitions: 0.7234
  Most common transition errors:
    NREM -> Wake: 45
    Wake -> NREM: 38
    NREM -> REM: 23
============================================================
```

---

## Transition Analysis

Sleep state transitions are the hardest epochs to score correctly. The diagnostics module provides specific analysis:

```python
from model.diagnostics import analyze_transition_errors

transition_acc, error_counts = analyze_transition_errors(y_true, y_pred)

print(f"Accuracy at transitions: {transition_acc:.3f}")
print("Most common errors:")
for error, count in sorted(error_counts.items(), key=lambda x: -x[1])[:5]:
    print(f"  {error}: {count}")
```

**Why this matters:**
- Researchers often focus on transition timing
- A model with 90% overall accuracy might have only 70% accuracy at transitions
- Understanding transition errors guides model improvement

---

## Recommended Validation Workflow

This is what `model/train.py` does; the function names below are the ones in that
file, so you can call them from a notebook or script.

### 1. Train/test split by time within each recording

```python
from model.train import split_train_test

train_df, test_df = split_train_test(
    df_features, train_size=0.75, group_col="ID_day", time_col="epoch"
)
```

The first 75% of each recording's epochs go to train, the last 25% to test.
Temporal order is kept and no recording is split randomly.

### 2. Hyperparameter search with grouped time-series CV

```python
from model.train import tune_hyperparams
from model.cross_validation import get_cv_splitter

cv = get_cv_splitter("group_time_series", n_splits=5, gap=6)
best_params, duration = tune_hyperparams(
    X_train, y_train_encoded, search_space, cv, groups=train_df["ID_day"],
    n_iter=50, random_state=42,
)
```

### 3. Train the final model on all labeled data

```python
from model.train import train_final_model

model, label_encoder, fit_time = train_final_model(X_all, y_all, best_params)
```

### 4. Evaluate with the full metric set

```python
from model.diagnostics import evaluate_model

metrics = evaluate_model(y_test, y_pred, class_labels=["Non REM", "REM", "Wake"])
print(metrics.summary())
```

The metrics are stored in the model's metadata (`test_cohen_kappa`,
`test_class_metrics`, ...) so every artifact carries its own report.

### 5. Leave-one-animal-out for the honest estimate

The holdout slice above shares animals with training. To estimate performance on
a *new* animal, use `get_cv_splitter("leave_one_out")` grouped by `ID_day`, or run
`python -m model.experiments.compare_feature_sets`, which does exactly that and
prints per-animal results. The numbers in `docs/MODEL.md` are LOO.

## Common Pitfalls

### ❌ Using Standard K-Fold CV

```python
# DON'T do this for time-series data
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5)  # Random splits!
```

### ✅ Use Time-Series Aware CV

```python
# DO this instead
from model.cross_validation import get_cv_splitter
cv = get_cv_splitter("group_time_series", gap=6)
scores = cross_val_score(model, X, y, cv=cv, groups=df['ID_day'])
```

### ❌ Only Reporting Accuracy

Accuracy hides poor performance on minority classes (REM).

### ✅ Report Multiple Metrics

Always report:
- Cohen's Kappa
- Per-class recall (especially REM)
- Confusion matrix
- Transition accuracy

---

## References

1. Cohen, J. (1960). A coefficient of agreement for nominal scales. *Educational and Psychological Measurement*, 20(1), 37-46.

2. Landis, J. R., & Koch, G. G. (1977). The measurement of observer agreement for categorical data. *Biometrics*, 33(1), 159-174.

3. Fonseca, P., et al. (2017). Validation of photoplethysmography-based sleep staging compared with polysomnography in healthy middle-aged adults. *Sleep*, 40(7).
