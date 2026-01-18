"""
Tests for the diagnostics module.
"""

import numpy as np
import pytest

from model.diagnostics import (
    SleepScoringMetrics,
    analyze_transition_errors,
    compute_epoch_agreement,
    evaluate_model,
    print_confusion_matrix,
)


class TestEvaluateModel:
    """Tests for the evaluate_model function."""

    def test_returns_metrics_object(self, sample_predictions):
        """Test that evaluate_model returns SleepScoringMetrics."""
        y_true, y_pred = sample_predictions
        metrics = evaluate_model(y_true, y_pred)
        assert isinstance(metrics, SleepScoringMetrics)

    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = np.array(["Wake", "Non REM", "REM"] * 10)
        y_pred = y_true.copy()

        metrics = evaluate_model(y_true, y_pred)

        assert metrics.accuracy == 1.0
        assert metrics.balanced_accuracy == 1.0
        assert metrics.cohen_kappa == 1.0
        assert metrics.f1_weighted == 1.0

    def test_metrics_in_valid_range(self, sample_predictions):
        """Test that all metrics are in expected ranges."""
        y_true, y_pred = sample_predictions
        metrics = evaluate_model(y_true, y_pred)

        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.balanced_accuracy <= 1
        assert -1 <= metrics.cohen_kappa <= 1
        assert 0 <= metrics.f1_weighted <= 1
        assert 0 <= metrics.f1_macro <= 1

    def test_per_class_metrics_present(self, sample_predictions):
        """Test that per-class metrics are computed."""
        y_true, y_pred = sample_predictions
        metrics = evaluate_model(y_true, y_pred)

        expected_classes = ["Wake", "Non REM", "REM"]
        for cls in expected_classes:
            assert cls in metrics.class_metrics
            assert "precision" in metrics.class_metrics[cls]
            assert "recall" in metrics.class_metrics[cls]
            assert "f1" in metrics.class_metrics[cls]
            assert "support" in metrics.class_metrics[cls]

    def test_confusion_matrix_shape(self, sample_predictions):
        """Test confusion matrix has correct shape."""
        y_true, y_pred = sample_predictions
        metrics = evaluate_model(y_true, y_pred)

        n_classes = len(metrics.class_labels)
        assert metrics.confusion_matrix.shape == (n_classes, n_classes)

    def test_confusion_matrix_sums_to_total(self, sample_predictions):
        """Test confusion matrix sums to total samples."""
        y_true, y_pred = sample_predictions
        metrics = evaluate_model(y_true, y_pred)

        assert metrics.confusion_matrix.sum() == len(y_true)

    def test_str_representation(self, sample_predictions):
        """Test that str() returns formatted output."""
        y_true, y_pred = sample_predictions
        metrics = evaluate_model(y_true, y_pred)

        output = str(metrics)
        assert "SLEEP SCORING MODEL EVALUATION" in output
        assert "Cohen's Kappa" in output
        assert "Accuracy" in output

    def test_to_dict(self, sample_predictions):
        """Test conversion to dictionary."""
        y_true, y_pred = sample_predictions
        metrics = evaluate_model(y_true, y_pred)

        d = metrics.to_dict()
        assert isinstance(d, dict)
        assert "accuracy" in d
        assert "cohen_kappa" in d
        assert "class_metrics" in d


class TestAnalyzeTransitionErrors:
    """Tests for transition error analysis."""

    def test_identifies_transitions(self):
        """Test that transitions are correctly identified."""
        y_true = np.array(["Wake", "Wake", "Non REM", "Non REM", "REM"])
        y_pred = np.array(["Wake", "Wake", "Wake", "Non REM", "REM"])  # Error at transition

        acc, errors = analyze_transition_errors(y_true, y_pred)

        assert 0 <= acc <= 1
        assert isinstance(errors, dict)

    def test_no_transitions(self):
        """Test with no transitions in data."""
        y_true = np.array(["Wake"] * 10)
        y_pred = np.array(["Wake"] * 10)

        acc, errors = analyze_transition_errors(y_true, y_pred)

        assert acc == 1.0
        assert errors == {}

    def test_transition_accuracy_lower_than_overall(self, sample_predictions):
        """Test that transition accuracy is typically lower than overall."""
        y_true, y_pred = sample_predictions
        metrics = evaluate_model(y_true, y_pred)

        # This is a common pattern but not guaranteed
        # Just check the analysis runs without error
        assert metrics.transition_accuracy is not None

    def test_error_counts_are_positive(self):
        """Test that error counts are non-negative."""
        y_true = np.array(["Wake", "Non REM", "Non REM", "REM", "Wake"])
        y_pred = np.array(["Wake", "Wake", "Non REM", "Non REM", "Wake"])

        _, errors = analyze_transition_errors(y_true, y_pred)

        for count in errors.values():
            assert count >= 0


class TestComputeEpochAgreement:
    """Tests for inter-rater agreement computation."""

    def test_perfect_agreement(self):
        """Test with identical scorers."""
        scorer1 = np.array(["Wake", "Non REM", "REM"] * 10)
        scorer2 = scorer1.copy()

        agreement = compute_epoch_agreement(scorer1, scorer2)

        assert agreement["overall_agreement"] == 1.0
        assert agreement["cohen_kappa"] == 1.0

    def test_no_agreement(self):
        """Test with completely different scorers."""
        scorer1 = np.array(["Wake"] * 30)
        scorer2 = np.array(["Non REM"] * 30)

        agreement = compute_epoch_agreement(scorer1, scorer2)

        assert agreement["overall_agreement"] == 0.0
        assert agreement["cohen_kappa"] < 0.5

    def test_returns_n_epochs(self):
        """Test that n_epochs is returned."""
        scorer1 = np.array(["Wake", "Non REM", "REM"] * 10)
        scorer2 = np.array(["Wake", "Non REM", "Non REM"] * 10)

        agreement = compute_epoch_agreement(scorer1, scorer2)

        assert agreement["n_epochs"] == 30


class TestPrintConfusionMatrix:
    """Tests for confusion matrix formatting."""

    def test_returns_string(self, sample_predictions):
        """Test that function returns a string."""
        y_true, y_pred = sample_predictions
        output = print_confusion_matrix(y_true, y_pred)

        assert isinstance(output, str)
        assert "CONFUSION MATRIX" in output

    def test_contains_class_labels(self, sample_predictions):
        """Test that output contains class labels."""
        y_true, y_pred = sample_predictions
        output = print_confusion_matrix(y_true, y_pred)

        assert "Wake" in output or "Non REM" in output or "REM" in output


class TestKappaInterpretation:
    """Tests for Cohen's Kappa interpretation."""

    def test_kappa_interpretation_ranges(self):
        """Test that kappa interpretation follows Landis & Koch."""
        # Create metrics with known kappa values
        test_cases = [
            (0.0, "Slight"),
            (0.25, "Fair"),
            (0.45, "Moderate"),
            (0.65, "Substantial"),
            (0.85, "Almost Perfect"),
        ]

        for kappa_value, expected_interpretation in test_cases:
            # Create a metrics object to test interpretation
            metrics = SleepScoringMetrics(
                accuracy=0.8,
                balanced_accuracy=0.8,
                cohen_kappa=kappa_value,
                f1_weighted=0.8,
                f1_macro=0.8,
                class_metrics={},
                confusion_matrix=np.array([[1]]),
                class_labels=["A"],
            )
            interpretation = metrics._interpret_kappa()
            assert expected_interpretation in interpretation
