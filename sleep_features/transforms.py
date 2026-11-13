"""
Low-level signal transforms for sleep-scoring feature engineering.

These are the numeric primitives (FFT power, smoothing, band power,
time-domain features). They are intentionally free of any dependency on the
backend or the model so this package can be imported by any consumer.

Moved verbatim from backend/app/ml/sleep_functions.py during the feature-store
extraction (Phase 1). Behavior is frozen by tests/unit/test_feature_transforms.py.
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.fft import rfft
from scipy.integrate import simpson


def robust_std(x: np.ndarray) -> float:
    """
    Robust standard-deviation estimate via the median absolute deviation.

    1.4826 * MAD is a consistent estimator of sigma for Gaussian data but is
    insensitive to outliers/artifacts. Falls back to the ordinary std (then 1.0)
    when the MAD is zero or non-finite, so callers never divide by zero.
    """
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    scale = 1.4826 * np.median(np.abs(x - med))
    if not np.isfinite(scale) or scale == 0:
        scale = float(np.std(x))
    return scale if (np.isfinite(scale) and scale != 0) else 1.0


def compute_power(
    data: pd.DataFrame,
    window: int = 10,
    samplerate: int = 1000,
    freq_limit: int = 50,
    taper: str = None,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Computes a power spectrum from raw eeg/emg data.

    Args:
        data: DataFrame with 'eeg' and 'emg' as columns
        window: Window of time (in seconds) for power spectrum computation. Default=10
        samplerate: Sampling rate in Hz. Default=1000
        freq_limit: Maximum frequency (in Hz) retained in the power spectrum. Default=50
        taper: Optional window applied before the FFT to reduce spectral leakage
            ('hann'/'hanning', 'hamming'). Default None (no taper) preserves the
            legacy behavior; the v2 feature path passes 'hann'.

    Returns:
        Tuple of (eeg_power, emg_power) dicts mapping epoch -> power spectrum array
    """
    array_len = window * samplerate
    freq_limit = freq_limit * 10
    dt = 1 / samplerate

    taper_window = None
    if taper in ("hann", "hanning"):
        taper_window = np.hanning(array_len)
    elif taper == "hamming":
        taper_window = np.hamming(array_len)
    elif taper is not None:
        raise ValueError(f"Unknown taper '{taper}'")

    eeg_power = {}
    emg_power = {}

    epoch = 0
    for value in range(0, len(data), array_len):
        start = value
        end = start + array_len

        # Select EEG and EMG data based on start and stop points
        EEG = data["eeg"][start:end]
        EMG = data["emg"][start:end]

        eeg_centered = np.array(EEG) - np.array(EEG).mean()
        emg_centered = np.array(EMG) - np.array(EMG).mean()
        if taper_window is not None and len(eeg_centered) == array_len:
            eeg_centered = eeg_centered * taper_window
            emg_centered = emg_centered * taper_window

        # Compute Fourier Transform
        eeg_xf = rfft(eeg_centered)
        emg_xf = rfft(emg_centered)

        # Compute power spectrum
        eeg_Sxx = (2 * dt**2 / window * (eeg_xf * eeg_xf.conj())).real
        eeg_Sxx = eeg_Sxx[0:freq_limit]

        emg_Sxx = (2 * dt**2 / window * (emg_xf * emg_xf.conj())).real
        emg_Sxx = emg_Sxx[0:freq_limit]

        eeg_power[epoch] = eeg_Sxx
        emg_power[epoch] = emg_Sxx

        epoch += 1

    return eeg_power, emg_power


def smooth_signal(
    eeg_power: Dict[int, np.ndarray],
    emg_power: Dict[int, np.ndarray],
    window_len: int = 4,
    smooth_type: str = "flat",
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Smooths the power spectrum using a moving window.

    Args:
        eeg_power: Dict mapping epochs to EEG power spectra
        emg_power: Dict mapping epochs to EMG power spectra
        window_len: Size of smoothing window. Bigger = more smoothing. Default=4
        smooth_type: Kind of smoothing function ('flat', 'hanning', 'hamming', 'bartlett', 'blackman')

    Returns:
        Tuple of (smoothed_eeg, smoothed_emg) dicts
    """
    smoothed_eeg = {}
    smoothed_emg = {}

    if window_len % 2 != 0:
        raise ValueError("window length must be an even integer")

    for epoch in eeg_power:
        x_eeg = eeg_power[epoch]
        x_emg = emg_power[epoch]
        if x_eeg.ndim != 1:
            raise ValueError("smooth only accepts 1 dimension arrays.")

        if x_eeg.size < window_len:
            break

        if window_len < 3:
            return x_eeg, x_emg

        if smooth_type not in ["flat", "hanning", "hamming", "bartlett", "blackman"]:
            raise ValueError(
                "Smooth_type must be one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
            )

        s_eeg = np.r_[
            x_eeg[window_len - 1 : 0 : -1], x_eeg, x_eeg[-2 : -window_len - 1 : -1]
        ]
        s_emg = np.r_[
            x_emg[window_len - 1 : 0 : -1], x_emg, x_emg[-2 : -window_len - 1 : -1]
        ]

        if smooth_type == "flat":  # Moving average
            w = np.ones(window_len, "d")
        else:
            w = getattr(np, smooth_type)(window_len)

        y_eeg = np.convolve(w / w.sum(), s_eeg, mode="valid")
        y_emg = np.convolve(w / w.sum(), s_emg, mode="valid")

        smoothed_eeg[epoch] = y_eeg[(int(window_len / 2) - 1) : -int(window_len / 2)]
        smoothed_emg[epoch] = y_emg[(int(window_len / 2) - 1) : -int(window_len / 2)]

    return smoothed_eeg, smoothed_emg


def compute_relative_power(
    smoothed_eeg: Dict[int, np.ndarray],
    freq_res: float = 0.1,
    delta_lower: float = 0.5,
    delta_upper: float = 4,
    theta_lower: float = 5.5,
    theta_upper: float = 8.5,
) -> pd.DataFrame:
    """
    Computes relative power for delta and theta bands.

    Args:
        smoothed_eeg: Dict mapping epochs to smoothed EEG power spectra
        freq_res: Frequency resolution of power spectrum. Default=0.1
        delta_lower/upper: Delta band limits (0.5-4 Hz)
        theta_lower/upper: Theta band limits (5.5-8.5 Hz)

    Returns:
        DataFrame with columns: delta_rel, theta_rel, theta_over_delta
    """
    rel_power = {}
    theta_length = ((theta_upper - theta_lower) / freq_res) + 1
    delta_length = ((delta_upper - delta_lower) / freq_res) + 1

    for epoch in smoothed_eeg:
        x = smoothed_eeg[epoch]
        rel_power_list = []

        power_axis = np.arange(len(x)) * freq_res
        idx_delta = np.logical_and(power_axis >= delta_lower, power_axis <= delta_upper)
        idx_theta = np.logical_and(power_axis >= theta_lower, power_axis <= theta_upper)

        if (len((np.where(idx_theta))[0])) >= theta_length and (
            len((np.where(idx_delta))[0])
        ) >= delta_length:
            # Compute total power
            total_power = simpson(x, dx=freq_res)

            # Compute delta power
            delta_power = (simpson(x[idx_delta], dx=freq_res)) / total_power
            rel_power_list.append(delta_power)

            # Compute theta power
            theta_power = (simpson(x[idx_theta], dx=freq_res)) / total_power
            rel_power_list.append(theta_power)

            # Compute proportion of theta to delta (guard against a zero/degenerate
            # delta band, which would otherwise produce inf/nan).
            theta_over_delta = theta_power / delta_power if delta_power > 0 else 0.0
            rel_power_list.append(theta_over_delta)

            rel_power[epoch] = rel_power_list
        else:
            rel_power[epoch] = [0, 0, 0]

    rel_power = pd.DataFrame.from_dict(rel_power).T
    rel_power.columns = ["delta_rel", "theta_rel", "theta_over_delta"]

    return rel_power


def compute_signal_features(
    data: pd.DataFrame, window: int = 10, samplerate: int = 1000, start_epoch: int = 9
) -> pd.DataFrame:
    """
    Computes features from EEG and EMG data for ML model.

    Args:
        data: DataFrame with 'eeg' and 'emg' columns
        window: Epoch window in seconds. Default=10
        samplerate: Sample rate in Hz. Default=1000
        start_epoch: Baseline epoch for normalization. Default=9

    Returns:
        DataFrame with signal feature columns per epoch
    """
    array_len = window * samplerate
    EEG_quantile_diff = {}
    EEG_quantile_80 = {}
    EEG_ptp = {}
    EEG_ss = {}
    EEG_amp = {}
    EEG_std = {}
    EMG_std = {}
    EMG_events = {}
    EMG_ptp = {}

    # Compute baseline EEG/EMG signals
    base_eeg_epoch = data["eeg"][
        (start_epoch * array_len) : ((start_epoch * array_len) + array_len)
    ]
    base_emg_epoch = data["emg"][
        (start_epoch * array_len) : ((start_epoch * array_len) + array_len)
    ]

    base_quantile_diff_EEG = np.sum(np.quantile(base_eeg_epoch, 0.8) - base_eeg_epoch)
    base_quantile_80_EEG = np.quantile(base_eeg_epoch, 0.8)
    base_ptp_EEG = np.ptp(base_eeg_epoch)
    base_ss_EEG = np.sum(np.square(base_eeg_epoch))
    base_std_EEG = np.std(base_eeg_epoch)
    base_amp_EEG = np.mean(np.absolute(base_eeg_epoch))
    base_std_EMG = np.std(base_emg_epoch)
    base_ptp_EMG = np.ptp(base_emg_epoch)

    event_threshold = 2 * base_std_EMG

    epoch = 0
    for value in range(0, len(data), array_len):
        start = value
        end = start + array_len

        # Select EEG and EMG data based on start and stop points
        EEG = data["eeg"][start:end]
        EMG = data["emg"][start:end]

        # Calculate features and add to dictionaries
        EEG_quantile_diff[epoch] = (
            np.sum(EEG - np.quantile(EEG, 0.8)) / base_quantile_diff_EEG
        )
        EEG_quantile_80[epoch] = np.quantile(EEG, 0.8) / base_quantile_80_EEG
        EEG_ptp[epoch] = np.ptp(EEG) / base_ptp_EEG
        EEG_ss[epoch] = np.sum(np.square(EEG)) / base_ss_EEG
        EEG_amp[epoch] = np.mean(np.absolute(EEG)) / base_amp_EEG
        EEG_std[epoch] = np.std(EEG) / base_std_EEG
        EMG_std[epoch] = np.std(EMG) / base_std_EMG
        EMG_ptp[epoch] = np.ptp(EMG) / base_ptp_EMG

        # Calculate EMG events above event_threshold
        event_array = EMG.loc[EMG > event_threshold]

        if not len(event_array) == 0:
            event_dict = {}
            count = 0
            first = event_array.index[0]

            for idx in event_array.index:
                if idx + 1 not in event_array.index:
                    last = idx
                    event_dict[count] = event_array.loc[first:last]
                    count += 1
                    first = last
            EMG_events[epoch] = len(event_dict)
        else:
            EMG_events[epoch] = 0

        epoch += 1

    output = pd.DataFrame(
        [
            EEG_quantile_diff,
            EEG_quantile_80,
            EEG_ptp,
            EEG_ss,
            EEG_amp,
            EEG_std,
            EMG_std,
            EMG_events,
            EMG_ptp,
        ]
    ).T
    output.columns = [
        "EEG_quantile_diff",
        "EEG_quantile_80",
        "EEG_ptp",
        "EEG_ss",
        "EEG_amp",
        "EEG_std",
        "EMG_std",
        "EMG_events",
        "EMG_ptp",
    ]

    return output


def compute_signal_features_raw(
    data: pd.DataFrame,
    window: int = 10,
    samplerate: int = 1000,
    emg_event_k: float = 2.0,
) -> pd.DataFrame:
    """
    Compute time-domain EEG/EMG features WITHOUT baseline-epoch normalization.

    This is the decoupled replacement for compute_signal_features: it emits raw
    per-epoch statistics and leaves normalization to a separate, pluggable stage
    (see sleep_features.normalize). It takes no ``start_epoch`` — there is no
    hand-picked baseline anywhere in this path.

    The EMG event threshold is derived from a robust statistic over the WHOLE
    recording (median + k * robust_std), so it is gain-consistent and does not
    depend on any single epoch.

    Args:
        data: DataFrame with 'eeg' and 'emg' columns
        window: Epoch window in seconds
        samplerate: Sample rate in Hz
        emg_event_k: Multiplier on the robust EMG scale for the event threshold

    Returns:
        DataFrame of raw (un-normalized) features, columns ordered to match
        config.GENERATED_FEATURE_COLS minus the spectral columns.
    """
    array_len = window * samplerate

    emg_all = data["emg"].to_numpy(dtype=float)
    event_threshold = float(np.median(emg_all)) + emg_event_k * robust_std(emg_all)

    rows = {
        "EEG_quantile_diff": {},
        "EEG_quantile_80": {},
        "EEG_ptp": {},
        "EEG_ss": {},
        "EEG_amp": {},
        "EEG_std": {},
        "EMG_std": {},
        "EMG_events": {},
        "EMG_ptp": {},
    }

    epoch = 0
    for value in range(0, len(data), array_len):
        EEG = data["eeg"].to_numpy(dtype=float)[value : value + array_len]
        EMG = data["emg"].to_numpy(dtype=float)[value : value + array_len]

        q80 = np.quantile(EEG, 0.8)
        rows["EEG_quantile_diff"][epoch] = np.sum(EEG - q80)
        rows["EEG_quantile_80"][epoch] = q80
        rows["EEG_ptp"][epoch] = np.ptp(EEG)
        rows["EEG_ss"][epoch] = np.sum(np.square(EEG))
        rows["EEG_amp"][epoch] = np.mean(np.absolute(EEG))
        rows["EEG_std"][epoch] = np.std(EEG)
        rows["EMG_std"][epoch] = np.std(EMG)
        rows["EMG_ptp"][epoch] = np.ptp(EMG)

        # Count contiguous runs of EMG above the recording-level threshold by
        # counting rising edges (0 -> above).
        above = (EMG > event_threshold).astype(int)
        rows["EMG_events"][epoch] = int(np.sum(np.diff(above, prepend=0) == 1))

        epoch += 1

    output = pd.DataFrame(rows)
    return output


def _hjorth(x: np.ndarray) -> Tuple[float, float]:
    """Return (mobility, complexity) Hjorth parameters for a 1-D signal.

    Both are ratios of successive-derivative variances and are therefore
    invariant to the signal's amplitude scale (gain).
    """
    x = np.asarray(x, dtype=float)
    dx = np.diff(x)
    ddx = np.diff(dx)
    var0 = np.var(x)
    var1 = np.var(dx)
    var2 = np.var(ddx)

    mobility = np.sqrt(var1 / var0) if var0 > 0 else 0.0
    mob_dx = np.sqrt(var2 / var1) if var1 > 0 else 0.0
    complexity = (mob_dx / mobility) if mobility > 0 else 0.0
    return float(mobility), float(complexity)


def compute_hjorth_features(
    data: pd.DataFrame, window: int = 10, samplerate: int = 1000
) -> pd.DataFrame:
    """Per-epoch Hjorth mobility/complexity for EEG, mobility for EMG.

    These are scale-invariant complexity descriptors that help separate the
    states (e.g. low-complexity NREM slow waves vs. higher-complexity REM/Wake).
    """
    array_len = window * samplerate
    eeg = data["eeg"].to_numpy(dtype=float)
    emg = data["emg"].to_numpy(dtype=float)

    rows = {
        "EEG_hjorth_mobility": {},
        "EEG_hjorth_complexity": {},
        "EMG_hjorth_mobility": {},
    }
    epoch = 0
    for value in range(0, len(data), array_len):
        eeg_mob, eeg_cmp = _hjorth(eeg[value : value + array_len])
        emg_mob, _ = _hjorth(emg[value : value + array_len])
        rows["EEG_hjorth_mobility"][epoch] = eeg_mob
        rows["EEG_hjorth_complexity"][epoch] = eeg_cmp
        rows["EMG_hjorth_mobility"][epoch] = emg_mob
        epoch += 1

    return pd.DataFrame(rows)
