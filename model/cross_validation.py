"""
Time-series aware cross-validation for sleep scoring.

Standard k-fold CV causes data leakage in time-series data because:
1. Adjacent epochs are highly correlated (sleep states persist for minutes)
2. Random shuffling mixes future data into training set

This module provides group-aware time-series CV that:
1. Respects temporal ordering within each recording (ID_day)
2. Prevents leakage between adjacent epochs via optional gap
3. Ensures each recording's data stays together or is split chronologically
"""

from typing import Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator


class GroupTimeSeriesSplit(BaseCrossValidator):
    """
    Time-series cross-validation that respects group boundaries.

    For sleep scoring, this ensures:
    - Each fold respects temporal order within recordings
    - No data leakage from future epochs to past
    - Optional gap between train/test to prevent adjacent-epoch leakage

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds for cross-validation.

    gap : int, default=0
        Number of epochs to exclude between train and test sets.
        Recommended: 6-12 epochs (1-2 minutes) to prevent autocorrelation leakage.

    test_size : float, default=0.2
        Proportion of each group to use for testing in each fold.

    Examples
    --------
    >>> cv = GroupTimeSeriesSplit(n_splits=5, gap=6)
    >>> for train_idx, test_idx in cv.split(X, y, groups=df['ID_day']):
    ...     X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    ...     # train model...
    """

    def __init__(
        self,
        n_splits: int = 5,
        gap: int = 0,
        test_size: float = 0.2,
    ):
        self.n_splits = n_splits
        self.gap = gap
        self.test_size = test_size

    def get_n_splits(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        groups: Optional[pd.Series] = None,
    ) -> int:
        """Return the number of splits."""
        return self.n_splits

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        groups: Optional[pd.Series] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices for each fold.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.

        y : pd.Series, optional
            Target variable (not used, for API compatibility).

        groups : pd.Series
            Group labels (e.g., ID_day). Each group is split independently
            to respect time-series ordering within recordings.

        Yields
        ------
        train_idx : np.ndarray
            Training set indices for this fold.

        test_idx : np.ndarray
            Test set indices for this fold.
        """
        if groups is None:
            raise ValueError("groups parameter is required for GroupTimeSeriesSplit")

        # Reset index to ensure alignment
        X = X.reset_index(drop=True)
        groups = groups.reset_index(drop=True)

        unique_groups = groups.unique()
        n_groups = len(unique_groups)

        # For each fold, we'll use an expanding window approach
        # Fold 1: train on first portion of each group, test on next portion
        # Fold 2: train on larger portion, test on next portion
        # etc.
        for fold_idx in range(self.n_splits):
            train_indices = []
            test_indices = []

            # Calculate train/test boundaries for this fold
            # Use expanding window: more data in later folds
            train_end_frac = (fold_idx + 1) / (self.n_splits + 1)
            test_start_frac = train_end_frac
            test_end_frac = min(1.0, train_end_frac + self.test_size)

            for group in unique_groups:
                group_mask = groups == group
                group_indices = np.where(group_mask)[0]
                n_samples = len(group_indices)

                if n_samples < 10:  # Skip very small groups
                    continue

                # Calculate split points for this group
                train_end = int(n_samples * train_end_frac)
                test_start = train_end + self.gap
                test_end = int(n_samples * test_end_frac)

                # Ensure valid indices
                train_end = max(1, train_end)
                test_start = min(test_start, n_samples - 1)
                test_end = min(test_end, n_samples)

                if test_start >= test_end:
                    continue

                # Add indices for this group
                train_indices.extend(group_indices[:train_end])
                test_indices.extend(group_indices[test_start:test_end])

            if len(train_indices) == 0 or len(test_indices) == 0:
                continue

            yield np.array(train_indices), np.array(test_indices)


class LeaveOneGroupOut(BaseCrossValidator):
    """
    Leave-one-recording-out cross-validation.

    Each fold trains on all recordings except one, and tests on that recording.
    Useful for assessing generalization to new subjects/recordings.

    This is stricter than GroupTimeSeriesSplit but may have high variance
    if recordings differ substantially.
    """

    def __init__(self):
        pass

    def get_n_splits(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        groups: Optional[pd.Series] = None,
    ) -> int:
        """Return the number of splits (one per unique group)."""
        if groups is None:
            raise ValueError("groups is required")
        return len(np.unique(groups))

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        groups: Optional[pd.Series] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices for leave-one-group-out CV.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.

        y : pd.Series, optional
            Target variable (not used).

        groups : pd.Series
            Group labels (e.g., ID_day).

        Yields
        ------
        train_idx, test_idx : np.ndarray
            Indices for train and test sets.
        """
        if groups is None:
            raise ValueError("groups parameter is required")

        groups = np.array(groups)
        unique_groups = np.unique(groups)

        for test_group in unique_groups:
            test_mask = groups == test_group
            train_mask = ~test_mask

            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]

            yield train_idx, test_idx


def get_cv_splitter(
    strategy: str = "group_time_series",
    n_splits: int = 5,
    gap: int = 6,
    test_size: float = 0.2,
) -> BaseCrossValidator:
    """
    Factory function to get the appropriate CV splitter.

    Parameters
    ----------
    strategy : str
        One of:
        - "group_time_series": Time-series aware CV within groups (recommended)
        - "leave_one_out": Leave-one-recording-out CV (stricter)

    n_splits : int
        Number of folds (for group_time_series only).

    gap : int
        Number of epochs gap between train/test (for group_time_series only).
        Recommended: 6 epochs (1 minute at 10s epochs) to prevent autocorrelation.

    test_size : float
        Proportion of each group for testing (for group_time_series only).

    Returns
    -------
    cv : BaseCrossValidator
        Configured cross-validation splitter.

    Examples
    --------
    >>> cv = get_cv_splitter("group_time_series", n_splits=5, gap=6)
    >>> from sklearn.model_selection import cross_val_score
    >>> scores = cross_val_score(model, X, y, cv=cv, groups=df['ID_day'])
    """
    if strategy == "group_time_series":
        return GroupTimeSeriesSplit(n_splits=n_splits, gap=gap, test_size=test_size)
    elif strategy == "leave_one_out":
        return LeaveOneGroupOut()
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'group_time_series' or 'leave_one_out'")
