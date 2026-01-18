# Testing Guide

This document explains how to run tests for the Mora sleep scoring project.

---

## Quick Start

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/unit/test_diagnostics.py

# Run specific test class
pytest tests/unit/test_diagnostics.py::TestEvaluateModel

# Run specific test
pytest tests/unit/test_diagnostics.py::TestEvaluateModel::test_perfect_predictions
```

---

## Test Structure

```
tests/
├── conftest.py                      # Shared fixtures
├── unit/
│   ├── test_cross_validation.py     # CV splitter tests
│   └── test_diagnostics.py          # Metrics/diagnostics tests
└── integration/
    └── test_backend_api.py          # FastAPI endpoint tests
```

---

## Test Categories

### Unit Tests (`tests/unit/`)

Fast, isolated tests for individual modules.

**Cross-validation tests:**
- Temporal order is preserved within groups
- No overlap between train/test sets
- Gap parameter creates proper separation
- Correct number of splits generated

**Diagnostics tests:**
- Metrics are in valid ranges
- Perfect predictions yield perfect scores
- Per-class metrics are computed
- Confusion matrix sums correctly
- Cohen's Kappa interpretation follows Landis & Koch

### Integration Tests (`tests/integration/`)

Tests that verify components work together.

**Backend API tests:**
- Health endpoint returns expected format
- Model info endpoint includes all fields
- Score endpoint accepts WAV files
- CORS headers are present

---

## Fixtures

Common test fixtures are defined in `tests/conftest.py`:

| Fixture | Description |
|---------|-------------|
| `sample_sleep_data` | DataFrame with synthetic sleep data (3 groups, 100 epochs each) |
| `feature_columns` | List of feature column names |
| `sample_predictions` | Tuple of (y_true, y_pred) with realistic sleep patterns |
| `sample_wav_data` | DataFrame with synthetic EEG/EMG signals |

**Usage:**
```python
def test_something(sample_sleep_data):
    # sample_sleep_data is automatically provided
    assert len(sample_sleep_data) == 300
```

---

## Running Backend Tests

Backend integration tests require the FastAPI app:

```bash
# From project root
cd backend
pip install -r requirements.txt
cd ..

# Run backend tests
pytest tests/integration/test_backend_api.py
```

Note: Some tests may skip if the model file isn't found.

---

## Coverage

To run tests with coverage:

```bash
# Install coverage
pip install pytest-cov

# Run with coverage
pytest --cov=model --cov=backend/app --cov-report=html

# Open coverage report
open htmlcov/index.html
```

---

## Adding New Tests

### Unit Test Template

```python
"""Tests for my_module."""

import pytest
from my_module import my_function


class TestMyFunction:
    """Tests for my_function."""

    def test_basic_case(self):
        """Test basic functionality."""
        result = my_function(input_value)
        assert result == expected_value

    def test_edge_case(self):
        """Test edge case handling."""
        with pytest.raises(ValueError):
            my_function(invalid_input)

    def test_with_fixture(self, sample_sleep_data):
        """Test using a fixture."""
        result = my_function(sample_sleep_data)
        assert result is not None
```

### Fixture Template

Add to `tests/conftest.py`:

```python
@pytest.fixture
def my_fixture():
    """Description of what this fixture provides."""
    # Setup
    data = create_test_data()

    yield data

    # Teardown (optional)
    cleanup()
```

---

## Continuous Integration

Tests run automatically on push via GitHub Actions (if configured).

To set up CI, create `.github/workflows/test.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r backend/requirements.txt
          pip install pytest pytest-cov
      - name: Run tests
        run: pytest --cov
```

---

## Troubleshooting

### Import Errors

If you get import errors, ensure you're running from the project root:

```bash
cd /path/to/sleep
pytest
```

### Model Not Found

Integration tests may fail if the model file is missing. Ensure:
- `backend/models/XGBoost_1.2.4.pkl` exists, OR
- `model_artifacts/XGBoost_1.2.4/XGBoost_1.2.4.pkl` exists

### Slow Tests

To run only fast unit tests:

```bash
pytest tests/unit/ -x  # Stop on first failure
```
