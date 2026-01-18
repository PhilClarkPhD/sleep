"""
Scoring service - orchestrates the complete scoring pipeline.
"""

import io
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.io import wavfile

from app.core.config import settings
from app.core.model_loader import ModelManager
from app.ml import sleep_functions as sleep
from app.schemas.scoring import EpochScore, SignalData, ScoringResponse, ScoringStats

logger = logging.getLogger(__name__)


@dataclass
class ScoringResult:
    """Internal result container for scoring."""

    epochs: List[EpochScore]
    summary: ScoringStats
    samplerate: int
    baseline_epoch: int
    signal_data: Optional[SignalData] = None


class ScoringService:
    """Handles the complete scoring pipeline."""

    # Number of samples per epoch for visualization (downsample from 10000 to 500)
    DISPLAY_SAMPLES_PER_EPOCH = 500

    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager

    def _downsample_signal(self, signal: np.ndarray, target_samples: int) -> List[float]:
        """Downsample a signal to target number of samples using decimation."""
        if len(signal) <= target_samples:
            return signal.tolist()
        # Use simple decimation (take every nth sample)
        step = len(signal) // target_samples
        return signal[::step][:target_samples].tolist()

    def extract_signal_data(
        self,
        df: pd.DataFrame,
        samplerate: int,
        start_epoch: int = 2,
    ) -> SignalData:
        """
        Extract signal data for visualization.

        Args:
            df: DataFrame with 'eeg' and 'emg' columns
            samplerate: Sample rate in Hz
            start_epoch: Baseline epoch for feature computation

        Returns:
            SignalData with downsampled signals and power spectra
        """
        epoch_duration = settings.EPOCH_DURATION_SECONDS
        samples_per_epoch = samplerate * epoch_duration
        n_epochs = len(df) // samples_per_epoch

        # Compute power spectra
        eeg_power, emg_power = sleep.compute_power(df, window=epoch_duration, samplerate=samplerate)
        smoothed_eeg, _ = sleep.smooth_signal(eeg_power, emg_power)
        rel_power = sleep.compute_relative_power(smoothed_eeg)

        # Extract signals per epoch
        eeg_data = []
        emg_data = []
        power_data = []
        delta_power = []
        theta_power = []

        for epoch in range(n_epochs):
            start_idx = epoch * samples_per_epoch
            end_idx = start_idx + samples_per_epoch

            # Downsample EEG and EMG signals
            eeg_epoch = df["eeg"].iloc[start_idx:end_idx].values
            emg_epoch = df["emg"].iloc[start_idx:end_idx].values

            eeg_data.append(self._downsample_signal(eeg_epoch, self.DISPLAY_SAMPLES_PER_EPOCH))
            emg_data.append(self._downsample_signal(emg_epoch, self.DISPLAY_SAMPLES_PER_EPOCH))

            # Power spectrum (already has reasonable resolution)
            if epoch in smoothed_eeg:
                power_data.append(smoothed_eeg[epoch].tolist())
            else:
                power_data.append([])

            # Relative power values
            if epoch in rel_power.index:
                delta_power.append(float(rel_power.loc[epoch, "delta_rel"]))
                theta_power.append(float(rel_power.loc[epoch, "theta_rel"]))
            else:
                delta_power.append(0.0)
                theta_power.append(0.0)

        # Time axis for display (in seconds within epoch)
        time_axis = np.linspace(0, epoch_duration, self.DISPLAY_SAMPLES_PER_EPOCH).tolist()

        # Frequency axis for power spectrum (0.1 Hz resolution, 0-50 Hz)
        freq_axis = np.arange(0, 50, 0.1).tolist()

        return SignalData(
            eeg=eeg_data,
            emg=emg_data,
            power_spectrum=power_data,
            time_axis=time_axis,
            freq_axis=freq_axis,
            delta_power=delta_power,
            theta_power=theta_power,
        )

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
        include_signals: bool = True,
    ) -> ScoringResult:
        """
        Score a WAV file.

        Args:
            file_bytes: Raw bytes of the WAV file
            start_epoch: Baseline epoch for normalization (should be Wake state)
            include_signals: Whether to include signal data for visualization

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

        # Step 8: Extract signal data if requested
        signal_data = None
        if include_signals:
            logger.info("Extracting signal data for visualization")
            signal_data = self.extract_signal_data(df, samplerate, start_epoch)

        return ScoringResult(
            epochs=epochs,
            summary=summary,
            samplerate=samplerate,
            baseline_epoch=start_epoch,
            signal_data=signal_data,
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
