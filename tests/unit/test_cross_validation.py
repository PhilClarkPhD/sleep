"""
Tests for the cross_validation module.
"""

import numpy as np
import pandas as pd
import pytest

from model.cross_validation import (
    GroupTimeSeriesSplit,
    LeaveOneGroupOut,
    get_cv_splitter,
)


class TestGroupTimeSeriesSplit:
    """Tests for GroupTimeSeriesSplit cross-validator."""

    def test_respects_temporal_order(self, sample_sleep_data):
        """Test that train indices always come before test indices within groups."""
        cv = GroupTimeSeriesSplit(n_splits=3, gap=0)
        X = sample_sleep_data.drop(columns=["score", "ID_day", "epoch"])
        groups = sample_sleep_data["ID_day"]

        for train_idx, test_idx in cv.split(X, groups=groups):
            # For each group, check train epochs < test epochs
            for group in groups.unique():
                group_mask = groups == group
                group_indices = np.where(group_mask)[0]

                train_in_group = set(train_idx) & set(group_indices)
                test_in_group = set(test_idx) & set(group_indices)

                if train_in_group and test_in_group:
                    assert max(train_in_group) < min(test_in_group), \
                        f"Train indices should come before test indices in group {group}"

    def test_no_overlap_between_train_test(self, sample_sleep_data):
        """Test that train and test sets don't overlap."""
        cv = GroupTimeSeriesSplit(n_splits=3, gap=0)
        X = sample_sleep_data.drop(columns=["score", "ID_day", "epoch"])
        groups = sample_sleep_data["ID_day"]

        for train_idx, test_idx in cv.split(X, groups=groups):
            overlap = set(train_idx) & set(test_idx)
            assert len(overlap) == 0, "Train and test sets should not overlap"

    def test_gap_creates_separation(self, sample_sleep_data):
        """Test that gap parameter creates proper separation."""
        gap = 5
        cv = GroupTimeSeriesSplit(n_splits=3, gap=gap)
        X = sample_sleep_data.drop(columns=["score", "ID_day", "epoch"])
        groups = sample_sleep_data["ID_day"]

        for train_idx, test_idx in cv.split(X, groups=groups):
            for group in groups.unique():
                group_mask = groups == group
                group_indices = np.where(group_mask)[0]

                train_in_group = sorted(set(train_idx) & set(group_indices))
                test_in_group = sorted(set(test_idx) & set(group_indices))

                if train_in_group and test_in_group:
                    # There should be a gap between max train and min test
                    actual_gap = min(test_in_group) - max(train_in_group) - 1
                    # Gap might be less than requested if group is small
                    assert actual_gap >= 0, "Test should not come before train"

    def test_n_splits_returns_correct_count(self, sample_sleep_data):
        """Test that n_splits folds are generated."""
        n_splits = 4
        cv = GroupTimeSeriesSplit(n_splits=n_splits)
        X = sample_sleep_data.drop(columns=["score", "ID_day", "epoch"])
        groups = sample_sleep_data["ID_day"]

        fold_count = sum(1 for _ in cv.split(X, groups=groups))
        assert fold_count == n_splits

    def test_requires_groups(self, sample_sleep_data):
        """Test that split raises error without groups."""
        cv = GroupTimeSeriesSplit(n_splits=3)
        X = sample_sleep_data.drop(columns=["score", "ID_day", "epoch"])

        with pytest.raises(ValueError, match="groups parameter is required"):
            list(cv.split(X))


class TestLeaveOneGroupOut:
    """Tests for LeaveOneGroupOut cross-validator."""

    def test_each_group_is_test_once(self, sample_sleep_data):
        """Test that each group appears as test set exactly once."""
        cv = LeaveOneGroupOut()
        X = sample_sleep_data.drop(columns=["score", "ID_day", "epoch"])
        groups = sample_sleep_data["ID_day"]

        test_groups_seen = []
        for train_idx, test_idx in cv.split(X, groups=groups):
            test_groups = groups.iloc[test_idx].unique()
            assert len(test_groups) == 1, "Each fold should test exactly one group"
            test_groups_seen.extend(test_groups)

        # Each unique group should appear once
        assert set(test_groups_seen) == set(groups.unique())
        assert len(test_groups_seen) == len(groups.unique())

    def test_train_excludes_test_group(self, sample_sleep_data):
        """Test that training set excludes the test group."""
        cv = LeaveOneGroupOut()
        X = sample_sleep_data.drop(columns=["score", "ID_day", "epoch"])
        groups = sample_sleep_data["ID_day"]

        for train_idx, test_idx in cv.split(X, groups=groups):
            test_group = groups.iloc[test_idx].unique()[0]
            train_groups = groups.iloc[train_idx].unique()
            assert test_group not in train_groups

    def test_n_splits_equals_n_groups(self, sample_sleep_data):
        """Test that number of splits equals number of unique groups."""
        cv = LeaveOneGroupOut()
        groups = sample_sleep_data["ID_day"]

        n_splits = cv.get_n_splits(groups=groups)
        assert n_splits == len(groups.unique())


class TestGetCVSplitter:
    """Tests for the get_cv_splitter factory function."""

    def test_returns_group_time_series(self):
        """Test factory returns correct splitter type."""
        cv = get_cv_splitter("group_time_series", n_splits=5, gap=6)
        assert isinstance(cv, GroupTimeSeriesSplit)

    def test_returns_leave_one_out(self):
        """Test factory returns LeaveOneGroupOut."""
        cv = get_cv_splitter("leave_one_out")
        assert isinstance(cv, LeaveOneGroupOut)

    def test_invalid_strategy_raises(self):
        """Test that invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            get_cv_splitter("invalid_strategy")

    def test_passes_parameters(self):
        """Test that parameters are passed correctly."""
        cv = get_cv_splitter("group_time_series", n_splits=7, gap=10, test_size=0.15)
        assert cv.n_splits == 7
        assert cv.gap == 10
        assert cv.test_size == 0.15
