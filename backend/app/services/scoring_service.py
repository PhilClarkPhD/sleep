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

import sleep_features as sleep
from sleep_features.qc import compute_qc_flags
from sleep_features.reliability import assess_recording, emg_dynamic_range

from app.core.config import settings
from app.core.model_loader import ModelManager
from app.schemas.scoring import (
    EpochScore,
    RecordingReliability,
    SignalData,
    ScoringResponse,
    ScoringStats,
)

# Label used for epochs that fail signal QC. Matches the lab's hand-scoring
# convention for artifact epochs and the frontend's SleepState type.
UNSCORED_LABEL = "Unscored"

logger = logging.getLogger(__name__)


@dataclass
class ScoringResult:
    """Internal result container for scoring."""

    epochs: List[EpochScore]
    unfiltered_epochs: List[EpochScore]
    summary: ScoringStats
    samplerate: int
    baseline_epoch: int
    reliability: Optional[RecordingReliability] = None
    signal_data: Optional[SignalData] = None


class ScoringService:
    """Handles the complete scoring pipeline."""

    # Number of samples per epoch for visualization (downsample from 10000 to 500)
    DISPLAY_SAMPLES_PER_EPOCH = 500

    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager

    def _model_feature_cols(self) -> List[str]:
        """Feature columns this model expects (artifact metadata wins)."""
        meta = self.model_manager.metadata or {}
        return meta.get("feature_cols") or settings.FEATURE_COLS

    def _qc_mask(self, df: pd.DataFrame, n: int, samplerate: int) -> List[bool]:
        """
        Per-epoch unscorable mask from signal QC only (Layer A: flat, clipped,
        non-finite, gross-amplitude artifacts). Model-agnostic, so it runs for
        every model. Distributional OOD (Layer B) is deliberately NOT applied
        per-epoch; see _assess_reliability.
        """
        qc = compute_qc_flags(df, samplerate=samplerate)
        flags = [bool(x) for x in qc["qc_flag"].to_numpy()[:n]]
        if len(flags) < n:
            flags += [False] * (n - len(flags))
        n_bad = sum(flags)
        if n_bad:
            logger.info(f"Signal QC flagged {n_bad}/{n} epochs as {UNSCORED_LABEL}")
        return flags

    def _assess_reliability(
        self, df: pd.DataFrame, features: pd.DataFrame, qc_flags: List[bool], samplerate: int
    ) -> RecordingReliability:
        """
        Recording-level reliability warnings: QC-fail rate, EMG dynamic range,
        and (when the model ships a Mahalanobis detector) the OOD rate. The OOD
        rate is a whole-recording signal: on a globally shifted animal 15-30% of
        epochs cross the per-epoch threshold while the model still scores them
        at ~0.9 accuracy, so blanking them would discard mostly-correct scores.
        """
        ood_flags = None
        detector = getattr(self.model_manager, "ood_detector", None)
        if detector is not None:
            cols = detector.feature_cols_ or self._model_feature_cols()
            ood_flags = detector.is_ood(features[cols].to_numpy())

        emg_range = emg_dynamic_range(
            df, window=settings.EPOCH_DURATION_SECONDS, samplerate=samplerate
        )
        report = assess_recording(qc_flags, emg_range, ood_flags)
        for w in report.warnings:
            logger.warning(f"Reliability [{w.code}]: {w.message}")
        d = report.to_dict()
        if not np.isfinite(d["emg_range_p05_p50"]):
            d["emg_range_p05_p50"] = None
        return RecordingReliability(**d)

    def _generate_features(
        self, df: pd.DataFrame, samplerate: int, start_epoch: int
    ) -> pd.DataFrame:
        """
        Generate the feature table using the pipeline the loaded model expects.

        A model artifact may declare its normalization strategy in metadata
        ("normalization": {"strategy": ...}). Models trained on the new
        feature store use automatic per-recording normalization (no baseline
        epoch); legacy models fall back to baseline-epoch normalization.
        """
        meta = self.model_manager.metadata or {}
        strategy = (meta.get("normalization") or {}).get("strategy")

        if strategy in ("robust_mad", "none", "baseline_epoch"):
            logger.info(f"Generating features via '{strategy}' pipeline")
            return sleep.generate_feature_table(
                df,
                normalization=strategy,
                samplerate=samplerate,
                start_epoch=start_epoch if strategy == "baseline_epoch" else None,
            )

        # Legacy model with no declared strategy: original baseline-epoch path.
        logger.info(f"Generating features via legacy path (baseline epoch={start_epoch})")
        return sleep.generate_features(df, start_epoch=start_epoch)

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

        # Step 2: Generate features (pipeline chosen by the model's metadata)
        metrics = self._generate_features(df, samplerate, start_epoch)

        # Step 3: Extract model features in correct order
        features = metrics[self._model_feature_cols()]

        # Step 4: Get predictions
        logger.info(f"Running predictions on {len(features)} epochs")
        predictions = self.model_manager.predict(features)

        # Step 5: Apply rule-based filter (on the model's state predictions)
        epoch_duration = settings.EPOCH_DURATION_SECONDS
        filtered_predictions = sleep.apply_rule_based_filter(predictions)

        # Step 6: Mark epochs that fail signal QC as Unscored (Layer A only).
        qc_flags = self._qc_mask(df, len(features), samplerate)
        predictions = list(predictions)
        filtered_predictions = list(filtered_predictions)
        for i, bad in enumerate(qc_flags):
            if bad:
                predictions[i] = UNSCORED_LABEL
                filtered_predictions[i] = UNSCORED_LABEL

        # Step 6b: Recording-level reliability warnings (QC rate, EMG contrast, OOD rate).
        reliability = self._assess_reliability(df, features, qc_flags, samplerate)

        unfiltered_epochs = [
            EpochScore(epoch=i, score=score, timestamp_seconds=i * epoch_duration)
            for i, score in enumerate(predictions)
        ]
        epochs = [
            EpochScore(epoch=i, score=score, timestamp_seconds=i * epoch_duration)
            for i, score in enumerate(filtered_predictions)
        ]

        # Step 7: Compute summary statistics (percentages over SCORABLE epochs)
        total = len(filtered_predictions)
        unscorable_count = filtered_predictions.count(UNSCORED_LABEL)
        scorable = total - unscorable_count
        wake_count = filtered_predictions.count("Wake")
        nrem_count = filtered_predictions.count("Non REM")
        rem_count = filtered_predictions.count("REM")

        duration_seconds = total * epoch_duration
        summary = ScoringStats(
            total_epochs=total,
            wake_count=wake_count,
            wake_percent=round(100 * wake_count / scorable, 2) if scorable > 0 else 0,
            nrem_count=nrem_count,
            nrem_percent=round(100 * nrem_count / scorable, 2) if scorable > 0 else 0,
            rem_count=rem_count,
            rem_percent=round(100 * rem_count / scorable, 2) if scorable > 0 else 0,
            recording_duration_seconds=duration_seconds,
            recording_duration_hours=round(duration_seconds / 3600, 2),
            unscorable_count=unscorable_count,
            unscorable_percent=round(100 * unscorable_count / total, 2) if total > 0 else 0,
        )

        logger.info(f"Scoring complete: {summary}")

        # Step 8: Extract signal data if requested
        signal_data = None
        if include_signals:
            logger.info("Extracting signal data for visualization")
            signal_data = self.extract_signal_data(df, samplerate, start_epoch)

        return ScoringResult(
            epochs=epochs,
            unfiltered_epochs=unfiltered_epochs,
            summary=summary,
            samplerate=samplerate,
            baseline_epoch=start_epoch,
            reliability=reliability,
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
        df, samplerate = self.load_wav_file(file_bytes)
        metrics = self._generate_features(df, samplerate, start_epoch)
        epochs = list(range(len(metrics)))
        return epochs, metrics
