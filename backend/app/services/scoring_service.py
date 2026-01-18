"""
Scoring service - orchestrates the complete scoring pipeline.
"""

import io
import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.io import wavfile

from app.core.config import settings
from app.core.model_loader import ModelManager
from app.ml import sleep_functions as sleep
from app.schemas.scoring import EpochScore, ScoringResponse, ScoringStats

logger = logging.getLogger(__name__)


@dataclass
class ScoringResult:
    """Internal result container for scoring."""

    epochs: List[EpochScore]
    summary: ScoringStats
    samplerate: int
    baseline_epoch: int


class ScoringService:
    """Handles the complete scoring pipeline."""

    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager

    def load_wav_file(self, file_bytes: bytes) -> Tuple[pd.DataFrame, int]:
        """
        Load a WAV file from bytes and extract EEG/EMG channels.

        Args:
            file_bytes: Raw bytes of the WAV file

        Returns:
            Tuple of (DataFrame with 'eeg' and 'emg' columns, samplerate)
        """
        # Read WAV file from bytes
        samplerate, data = wavfile.read(io.BytesIO(file_bytes))

        logger.info(f"Loaded WAV: samplerate={samplerate}, shape={data.shape}")

        # Handle stereo (2 channels) - Channel 0 = EEG, Channel 1 = EMG
        if len(data.shape) == 2 and data.shape[1] == 2:
            eeg = data[:, 0].astype(np.float64)
            emg = data[:, 1].astype(np.float64)
        elif len(data.shape) == 1:
            # Mono file - treat as EEG only, create zero EMG
            logger.warning("Mono WAV file detected. Using as EEG with zero EMG.")
            eeg = data.astype(np.float64)
            emg = np.zeros_like(eeg)
        else:
            raise ValueError(f"Unexpected WAV shape: {data.shape}. Expected stereo (2 channels).")

        df = pd.DataFrame({"eeg": eeg, "emg": emg})

        return df, samplerate

    def score(
        self,
        file_bytes: bytes,
        start_epoch: int = 2,
    ) -> ScoringResult:
        """
        Score a WAV file.

        Args:
            file_bytes: Raw bytes of the WAV file
            start_epoch: Baseline epoch for normalization (should be Wake state)

        Returns:
            ScoringResult with epochs, summary stats, and metadata
        """
        # Step 1: Load WAV file
        df, samplerate = self.load_wav_file(file_bytes)

        # Step 2: Generate features
        logger.info(f"Generating features with baseline epoch={start_epoch}")
        metrics = sleep.generate_features(df, start_epoch=start_epoch)

        # Step 3: Extract model features in correct order
        features = metrics[settings.FEATURE_COLS]

        # Step 4: Get predictions
        logger.info(f"Running predictions on {len(features)} epochs")
        predictions = self.model_manager.predict(features)

        # Step 5: Apply rule-based filter
        filtered_predictions = sleep.apply_rule_based_filter(predictions)

        # Step 6: Build epoch list with timestamps
        epoch_duration = settings.EPOCH_DURATION_SECONDS
        epochs = []
        for i, score in enumerate(filtered_predictions):
            epochs.append(
                EpochScore(
                    epoch=i,
                    score=score,
                    timestamp_seconds=i * epoch_duration,
                )
            )

        # Step 7: Compute summary statistics
        total = len(filtered_predictions)
        wake_count = filtered_predictions.count("Wake")
        nrem_count = filtered_predictions.count("Non REM")
        rem_count = filtered_predictions.count("REM")

        duration_seconds = total * epoch_duration
        summary = ScoringStats(
            total_epochs=total,
            wake_count=wake_count,
            wake_percent=round(100 * wake_count / total, 2) if total > 0 else 0,
            nrem_count=nrem_count,
            nrem_percent=round(100 * nrem_count / total, 2) if total > 0 else 0,
            rem_count=rem_count,
            rem_percent=round(100 * rem_count / total, 2) if total > 0 else 0,
            recording_duration_seconds=duration_seconds,
            recording_duration_hours=round(duration_seconds / 3600, 2),
        )

        logger.info(f"Scoring complete: {summary}")

        return ScoringResult(
            epochs=epochs,
            summary=summary,
            samplerate=samplerate,
            baseline_epoch=start_epoch,
        )

    def extract_features(
        self,
        file_bytes: bytes,
        start_epoch: int = 2,
    ) -> Tuple[List[int], pd.DataFrame]:
        """
        Extract features without scoring.

        Args:
            file_bytes: Raw bytes of the WAV file
            start_epoch: Baseline epoch for normalization

        Returns:
            Tuple of (epoch list, features DataFrame)
        """
        df, _ = self.load_wav_file(file_bytes)
        metrics = sleep.generate_features(df, start_epoch=start_epoch)
        epochs = list(range(len(metrics)))
        return epochs, metrics
