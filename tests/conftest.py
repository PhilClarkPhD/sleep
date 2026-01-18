"""
Pytest fixtures for Mora sleep scoring tests.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_sleep_data():
    """Generate sample sleep scoring data for testing."""
    np.random.seed(42)
    n_epochs = 100
    n_groups = 3

    data = []
    for group_id in range(n_groups):
        for epoch in range(n_epochs):
            # Generate random features
            row = {
                "ID_day": f"subject_{group_id}",
                "epoch": epoch,
                "EEG_quantile_80": np.random.randn(),
                "EEG_ptp": np.random.randn(),
                "EEG_ss": np.random.randn(),
                "EMG_std": np.random.randn(),
                "EMG_events": np.random.randint(0, 10),
                "EMG_ptp": np.random.randn(),
                "delta_rel": np.random.rand(),
                "theta_rel": np.random.rand(),
                "theta_over_delta": np.random.rand() * 2,
            }
            # Assign sleep state with some temporal structure
            if epoch < 20:
                row["score"] = "Wake"
            elif epoch < 60:
                row["score"] = "Non REM"
            elif epoch < 80:
                row["score"] = "REM" if np.random.rand() > 0.3 else "Non REM"
            else:
                row["score"] = "Wake"
            data.append(row)

    return pd.DataFrame(data)


@pytest.fixture
def feature_columns():
    """List of feature columns used by the model."""
    return [
        "EEG_quantile_80",
        "EEG_ptp",
        "EEG_ss",
        "EMG_std",
        "EMG_events",
        "EMG_ptp",
        "delta_rel",
        "theta_rel",
        "theta_over_delta",
    ]


@pytest.fixture
def sample_predictions():
    """Sample true and predicted labels for testing metrics."""
    np.random.seed(42)
    n = 200

    # Generate realistic sleep sequence
    true_labels = []
    states = ["Wake", "Non REM", "REM"]
    current_state = "Wake"
    state_duration = 0

    for _ in range(n):
        true_labels.append(current_state)
        state_duration += 1

        # Transition logic
        if current_state == "Wake" and state_duration > 10:
            if np.random.rand() > 0.7:
                current_state = "Non REM"
                state_duration = 0
        elif current_state == "Non REM" and state_duration > 15:
            if np.random.rand() > 0.8:
                current_state = np.random.choice(["REM", "Wake"])
                state_duration = 0
        elif current_state == "REM" and state_duration > 8:
            if np.random.rand() > 0.7:
                current_state = "Non REM"
                state_duration = 0

    # Generate predictions with ~85% accuracy
    pred_labels = []
    for label in true_labels:
        if np.random.rand() > 0.15:
            pred_labels.append(label)
        else:
            other = [s for s in states if s != label]
            pred_labels.append(np.random.choice(other))

    return np.array(true_labels), np.array(pred_labels)


@pytest.fixture
def sample_wav_data():
    """Generate sample WAV-like data for testing."""
    np.random.seed(42)
    samplerate = 1000
    duration_seconds = 100
    n_samples = samplerate * duration_seconds

    # Simulated EEG and EMG signals
    eeg = np.random.randn(n_samples) * 100
    emg = np.random.randn(n_samples) * 50

    return pd.DataFrame({"eeg": eeg, "emg": emg})
