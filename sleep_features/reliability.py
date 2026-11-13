"""
Recording-level reliability assessment — label-free warnings about whether a
whole recording is likely to score well.

Why recording-level and not per-epoch: the Mahalanobis OOD detector (Layer B)
answers "is this epoch unlike the training data". On a recording with a global
shift (e.g. an EMG channel with low gain), 15-30% of epochs cross the q=0.999
line even though the model still scores them at ~0.9 accuracy. Blanking those
epochs throws away mostly-correct scores. So we keep per-epoch "Unscored" for
signal-quality artifacts only (Layer A, sleep_features.qc) and report the OOD
rate, QC-fail rate, and EMG dynamic range as recording-level warnings.

Thresholds were chosen on a 10-animal cohort (docs/REM_RECALL_EXPERIMENT.md,
"Can we flag hard-to-score recordings before scoring?", 2026-10-26) and should be
revisited as the cohort grows:
    QC fail  > 1%   -> noisy recording (ZZ27, ZZ30)
    EMG p05/p50 > 0.6 -> weak EMG atonia contrast, expect missed REM (ZZ05)
    OOD      > 10%  -> recording unlike training data (ZZ05, ZZ27, ZZ30)
A weak-theta recording (ZZ01) is NOT detectable with these metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional

import numpy as np
import pandas as pd

from .config import EPOCH_DURATION

# Warning thresholds (recording-level).
QC_FAIL_WARN_PERCENT = 1.0
EMG_RANGE_WARN_RATIO = 0.6
OOD_WARN_PERCENT = 10.0


@dataclass
class ReliabilityWarning:
    code: str      # 'noisy' | 'weak_emg_contrast' | 'out_of_distribution'
    metric: str    # human-readable metric name
    value: float
    threshold: float
    message: str


@dataclass
class ReliabilityReport:
    """Recording-level scorability summary."""

    qc_fail_percent: float
    emg_range_p05_p50: float
    ood_percent: Optional[float]   # None when no OOD detector is available
    warnings: List[ReliabilityWarning] = field(default_factory=list)

    @property
    def level(self) -> str:
        return "warning" if self.warnings else "ok"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["level"] = self.level
        return d


def emg_dynamic_range(
    data: pd.DataFrame, window: int = EPOCH_DURATION, samplerate: int = 1000
) -> float:
    """
    Ratio of the 5th-percentile to the median of per-epoch raw EMG std.

    Quiet epochs (NREM/REM atonia) should have much lower EMG than a typical
    epoch, so a healthy recording sits well below 0.5. A ratio near 1 means the
    EMG barely changes across states (low gain or poor electrode) and the model
    cannot use EMG to separate REM from Wake.
    """
    array_len = window * samplerate
    emg = data["emg"].to_numpy(dtype=float)
    n_epochs = len(emg) // array_len
    if n_epochs == 0:
        return float("nan")
    stds = emg[: n_epochs * array_len].reshape(n_epochs, array_len).std(axis=1)
    p50 = np.percentile(stds, 50)
    if p50 <= 0:
        return float("nan")
    return float(np.percentile(stds, 5) / p50)


def assess_recording(
    qc_flags,
    emg_range: float,
    ood_flags=None,
    qc_warn: float = QC_FAIL_WARN_PERCENT,
    emg_warn: float = EMG_RANGE_WARN_RATIO,
    ood_warn: float = OOD_WARN_PERCENT,
) -> ReliabilityReport:
    """
    Build a ReliabilityReport from per-epoch flags and the EMG range metric.

    Args:
        qc_flags:  iterable of bool, Layer A per-epoch QC failures
        emg_range: output of emg_dynamic_range()
        ood_flags: iterable of bool from MahalanobisOOD.is_ood, or None
    """
    qc_arr = np.asarray(list(qc_flags), dtype=bool)
    qc_pct = float(100 * qc_arr.mean()) if qc_arr.size else 0.0

    ood_pct: Optional[float] = None
    if ood_flags is not None:
        ood_arr = np.asarray(list(ood_flags), dtype=bool)
        ood_pct = float(100 * ood_arr.mean()) if ood_arr.size else 0.0

    warnings: List[ReliabilityWarning] = []
    if qc_pct > qc_warn:
        warnings.append(ReliabilityWarning(
            code="noisy",
            metric="QC fail %",
            value=round(qc_pct, 2),
            threshold=qc_warn,
            message=(f"{qc_pct:.1f}% of epochs failed signal QC (threshold {qc_warn:g}%). "
                     "Noisy recording; review flagged epochs."),
        ))
    if np.isfinite(emg_range) and emg_range > emg_warn:
        warnings.append(ReliabilityWarning(
            code="weak_emg_contrast",
            metric="EMG range p05/p50",
            value=round(float(emg_range), 3),
            threshold=emg_warn,
            message=(f"EMG dynamic range {emg_range:.2f} (threshold {emg_warn:g}). "
                     "Weak atonia contrast; expect missed REM."),
        ))
    if ood_pct is not None and ood_pct > ood_warn:
        warnings.append(ReliabilityWarning(
            code="out_of_distribution",
            metric="OOD %",
            value=round(ood_pct, 2),
            threshold=ood_warn,
            message=(f"{ood_pct:.1f}% of epochs are far from the training distribution "
                     f"(threshold {ood_warn:g}%). Recording unlike training data; "
                     "review scores carefully."),
        ))

    return ReliabilityReport(
        qc_fail_percent=round(qc_pct, 3),
        emg_range_p05_p50=round(float(emg_range), 4) if np.isfinite(emg_range) else float("nan"),
        ood_percent=round(ood_pct, 3) if ood_pct is not None else None,
        warnings=warnings,
    )
