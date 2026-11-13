# Guide: run the tests

Run from the repo root with the venv active (Mac: `source .venv/bin/activate`;
Windows: `.\.venv\Scripts\Activate.ps1`). `requirements.txt` already includes
`pytest` and `httpx`.

```bash
pytest -q                       # everything: ~90 tests, ~10 seconds
pytest tests/unit -q            # feature math, normalization, QC/OOD, reliability, CV, diagnostics
pytest tests/integration -q     # starts the FastAPI app in-process and hits the endpoints with the real model
pytest tests/unit/test_reliability.py -v         # one file, verbose
pytest -k "reliability and emg" -v               # by name
```

Success looks like `88 passed` (count grows over time; what matters is 0 failed).
`pytest.ini` sets `-v --tb=short` and silences deprecation warnings.

## What is covered

| File | Covers |
|---|---|
| `tests/unit/test_feature_transforms.py` | FFT power, smoothing, relative power, time-domain features |
| `tests/unit/test_feature_set_v2.py` | v2 columns exist, are finite, respond to synthetic signals |
| `tests/unit/test_normalization.py` | `robust_mad` / `none` / `baseline_epoch` behaviour and which columns are touched |
| `tests/unit/test_qc_ood.py` | signal QC flags (flat, clipping, non-finite), Mahalanobis OOD fit/predict/save |
| `tests/unit/test_reliability.py` | EMG dynamic range metric, recording-level warnings and thresholds |
| `tests/unit/test_scores.py` | label-file parsing (simple and LabChart layouts), alignment, Unscored |
| `tests/unit/test_build.py` | recording discovery, `labels.json` overrides and exclusion, feature-store build + manifest on synthetic data |
| `tests/unit/test_cross_validation.py` | `GroupTimeSeriesSplit` and leave-one-group-out produce valid, non-leaking folds |
| `tests/unit/test_diagnostics.py` | metrics dataclass, transition analysis, agreement |
| `tests/integration/test_backend_api.py` | `/health`, `/model/info`, `/score` with a synthetic 30 s WAV, auth field, CORS |
| `tests/golden/generate_features_sample.csv` | Pinned v1 feature values for a fixed input: fails if feature math changes (deliberate guard; bump `FEATURE_CODE_VERSION` MAJOR if the change is intended) |

Not covered: the frontend (no test suite; test by hand per `EDIT_THE_UI.md`), the
training script end-to-end (too slow; `test_build.py` covers the data side), the
experiment scripts.

## Fixtures (`tests/conftest.py`)

`sample_sleep_data` (synthetic feature table with labels), `feature_columns`,
`sample_predictions`, `sample_wav_data` (synthetic stereo signal). Use them rather
than reading `training_data/`, which is not available on every machine.

## Writing a test

Put it in the matching file above, or a new `tests/unit/test_<topic>.py`. Pattern:

```python
import numpy as np, pandas as pd
from sleep_features.reliability import emg_dynamic_range

def test_emg_range_is_low_when_emg_varies():
    rng = np.random.default_rng(0)
    emg = np.concatenate([rng.normal(0, s, 10_000) for s in [5, 50] * 10])
    df = pd.DataFrame({"eeg": rng.normal(0, 100, len(emg)), "emg": emg})
    assert emg_dynamic_range(df) < 0.3
```

Build synthetic signals; never depend on real recordings. A 10-epoch, 1000 Hz
recording is 100k samples and runs in milliseconds.

## Gotchas

- If `MORA_API_KEYS` is set in your shell, integration tests fail with 401. Unset it.
- Integration tests load the real model from `backend/models/`; if that folder is
  empty they fail rather than skip.
- Import errors like `No module named app`: run from the repo root, not `tests/`.
