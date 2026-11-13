"""
Out-of-distribution (OOD) detection — Layer B of unscorable-data detection.

A tree classifier will confidently assign Wake/NREM/REM to epochs that look
nothing like its training data. This detector measures how far an epoch's
features sit from the training feature distribution (squared Mahalanobis
distance) and flags epochs beyond a chi-square threshold as OOD ("Unscorable").

It is *fit* at training time on the training feature distribution and saved with
the model artifact, but the class itself is dependency-light and lives in
sleep_features so the serving backend can deserialize and apply it without
importing the training package. The fitted instance is consumed against the same
feature columns the model uses.
"""

from __future__ import annotations

import joblib
import numpy as np
from scipy.stats import chi2


class MahalanobisOOD:
    """Mahalanobis-distance novelty detector fit on training features."""

    def __init__(self, quantile: float = 0.999):
        self.quantile = quantile
        self.mean_ = None
        self.inv_cov_ = None
        self.threshold_ = None
        self.feature_cols_ = None

    def fit(self, X, feature_cols=None) -> "MahalanobisOOD":
        arr = np.asarray(X, dtype=float)
        self.feature_cols_ = list(feature_cols) if feature_cols is not None else None
        self.mean_ = arr.mean(axis=0)
        cov = np.cov(arr, rowvar=False)
        # Ridge for numerical stability / singular covariance.
        cov = np.atleast_2d(cov) + 1e-6 * np.eye(arr.shape[1])
        self.inv_cov_ = np.linalg.pinv(cov)
        self.threshold_ = float(chi2.ppf(self.quantile, df=arr.shape[1]))
        return self

    def distance(self, X) -> np.ndarray:
        """Squared Mahalanobis distance for each row."""
        if self.mean_ is None:
            raise RuntimeError("Detector not fit")
        arr = np.asarray(X, dtype=float)
        centered = arr - self.mean_
        return np.einsum("ij,jk,ik->i", centered, self.inv_cov_, centered)

    def is_ood(self, X) -> np.ndarray:
        """Boolean mask: True where the epoch is out-of-distribution."""
        return self.distance(X) > self.threshold_

    def save(self, path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "MahalanobisOOD":
        return joblib.load(path)
