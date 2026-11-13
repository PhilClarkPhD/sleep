"""
Raw-signal I/O for the feature store.

Loads a stereo WAV (channel 0 = EEG, channel 1 = EMG) into the eeg/emg
DataFrame the transforms expect. Kept here (not in the backend) so every
consumer shares one loader.
"""

import io
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd
from scipy.io import wavfile


def load_wav(source: Union[str, Path, bytes]) -> Tuple[pd.DataFrame, int]:
    """
    Load EEG/EMG from a stereo WAV file path or raw bytes.

    Returns:
        (DataFrame with 'eeg' and 'emg' float64 columns, samplerate)
    """
    if isinstance(source, (bytes, bytearray)):
        samplerate, data = wavfile.read(io.BytesIO(source))
    else:
        samplerate, data = wavfile.read(str(source))

    if data.ndim == 2 and data.shape[1] >= 2:
        eeg = data[:, 0].astype(np.float64)
        emg = data[:, 1].astype(np.float64)
    elif data.ndim == 1:
        # Mono: treat as EEG with zero EMG (matches backend behavior).
        eeg = data.astype(np.float64)
        emg = np.zeros_like(eeg)
    else:
        raise ValueError(f"Unexpected WAV shape {data.shape}; expected stereo.")

    return pd.DataFrame({"eeg": eeg, "emg": emg}), int(samplerate)
