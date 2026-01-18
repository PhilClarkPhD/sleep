"""
Diagnostics and evaluation metrics for sleep scoring models.

This module provides comprehensive metrics beyond simple accuracy, including:
- Cohen's Kappa (standard in sleep scoring literature)
- Balanced accuracy (handles class imbalance)
- Per-class precision, recall, F1 (especially important for REM)
- Confusion matrix analysis
- Transition analysis (errors at state boundaries)
- Epoch-by-epoch agreement statistics

References:
- Cohen, J. (1960). A coefficient of agreement for nominal scales.
- Landis & Koch (1977). Kappa interpretation guidelines.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


@dataclass
class SleepScoringMetrics:
    """Container for all sleep scoring evaluation metrics."""

    # Overall metrics
    accuracy: float
    balanced_accuracy: float
    cohen_kappa: float
    f1_weighted: float
    f1_macro: float

    # Per-class metrics
    class_metrics: Dict[str, Dict[str, float]]

    # Confusion matrix
    confusion_matrix: np.ndarray
    class_labels: List[str]

    # Transition metrics
    transition_accuracy: Optional[float] = None
    transition_errors: Optional[Dict[str, int]] = None

    def __str__(self) -> str:
        """Human-readable summary of metrics."""
        lines = [
            "=" * 60,
            "SLEEP SCORING MODEL EVALUATION",
            "=" * 60,
            "",
            "OVERALL METRICS:",
            f"  Accuracy:          {self.accuracy:.4f}",
            f"  Balanced Accuracy: {self.balanced_accuracy:.4f}",
            f"  Cohen's Kappa:     {self.cohen_kappa:.4f}  {self._interpret_kappa()}",
            f"  F1 (weighted):     {self.f1_weighted:.4f}",
            f"  F1 (macro):        {self.f1_macro:.4f}",
            "",
            "PER-CLASS METRICS:",
        ]

        for class_name, metrics in self.class_metrics.items():
            lines.append(f"  {class_name}:")
            lines.append(f"    Precision: {metrics['precision']:.4f}")
            lines.append(f"    Recall:    {metrics['recall']:.4f}")
            lines.append(f"    F1:        {metrics['f1']:.4f}")
            lines.append(f"    Support:   {metrics['support']:.0f}")

        if self.transition_accuracy is not None:
            lines.extend([
                "",
                "TRANSITION ANALYSIS:",
                f"  Accuracy at transitions: {self.transition_accuracy:.4f}",
            ])
            if self.transition_errors:
                lines.append("  Most common transition errors:")
                sorted_errors = sorted(
                    self.transition_errors.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                for error, count in sorted_errors:
                    lines.append(f"    {error}: {count}")

        lines.extend(["", "=" * 60])
        return "\n".join(lines)

    def _interpret_kappa(self) -> str:
        """Interpret Cohen's Kappa according to Landis & Koch (1977)."""
        k = self.cohen_kappa
        if k < 0:
            return "(Poor)"
        elif k < 0.20:
            return "(Slight)"
        elif k < 0.40:
            return "(Fair)"
        elif k < 0.60:
            return "(Moderate)"
        elif k < 0.80:
            return "(Substantial)"
        else:
            return "(Almost Perfect)"

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "cohen_kappa": self.cohen_kappa,
            "f1_weighted": self.f1_weighted,
            "f1_macro": self.f1_macro,
            "class_metrics": self.class_metrics,
            "confusion_matrix": self.confusion_matrix.tolist(),
            "class_labels": self.class_labels,
            "transition_accuracy": self.transition_accuracy,
            "transition_errors": self.transition_errors,
        }


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: Optional[List[str]] = None,
    analyze_transitions: bool = True,
) -> SleepScoringMetrics:
    """
    Compute comprehensive evaluation metrics for sleep scoring.

    Parameters
    ----------
    y_true : array-like
        Ground truth sleep stage labels.

    y_pred : array-like
        Predicted sleep stage labels.

    class_labels : list of str, optional
        Class names in order. If None, inferred from data.

    analyze_transitions : bool, default=True
        Whether to analyze errors at state transitions.

    Returns
    -------
    metrics : SleepScoringMetrics
        Container with all computed metrics.

    Examples
    --------
    >>> metrics = evaluate_model(y_test, y_pred)
    >>> print(metrics)  # Human-readable summary
    >>> print(f"Kappa: {metrics.cohen_kappa:.3f}")
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if class_labels is None:
        class_labels = sorted(list(set(y_true) | set(y_pred)))

    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    f1_macro = f1_score(y_true, y_pred, average="macro")

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=class_labels, zero_division=0
    )

    class_metrics = {}
    for i, label in enumerate(class_labels):
        class_metrics[label] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": float(support[i]),
        }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)

    # Transition analysis
    transition_acc = None
    transition_errors = None
    if analyze_transitions:
        transition_acc, transition_errors = analyze_transition_errors(y_true, y_pred)

    return SleepScoringMetrics(
        accuracy=accuracy,
        balanced_accuracy=balanced_acc,
        cohen_kappa=kappa,
        f1_weighted=f1_weighted,
        f1_macro=f1_macro,
        class_metrics=class_metrics,
        confusion_matrix=cm,
        class_labels=class_labels,
        transition_accuracy=transition_acc,
        transition_errors=transition_errors,
    )


def analyze_transition_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[float, Dict[str, int]]:
    """
    Analyze model errors at sleep state transitions.

    Sleep state transitions are particularly challenging to score correctly.
    This function identifies epochs where the true state changed and evaluates
    model performance at those boundaries.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.

    y_pred : array-like
        Predicted labels.

    Returns
    -------
    transition_accuracy : float
        Accuracy specifically at transition epochs.

    error_counts : dict
        Count of each type of error at transitions.
        Keys are "true_transition -> predicted" format.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Find transition points (where true label changes)
    transitions = np.where(y_true[1:] != y_true[:-1])[0] + 1

    if len(transitions) == 0:
        return 1.0, {}

    # Include epoch before and after transition
    transition_epochs = set()
    for t in transitions:
        if t > 0:
            transition_epochs.add(t - 1)
        transition_epochs.add(t)
        if t < len(y_true) - 1:
            transition_epochs.add(t + 1)

    transition_epochs = sorted(list(transition_epochs))

    # Calculate accuracy at transitions
    y_true_trans = y_true[transition_epochs]
    y_pred_trans = y_pred[transition_epochs]
    transition_accuracy = accuracy_score(y_true_trans, y_pred_trans)

    # Categorize errors
    error_counts = {}
    for i, t_idx in enumerate(transition_epochs):
        if y_true[t_idx] != y_pred[t_idx]:
            error_key = f"{y_true[t_idx]} -> {y_pred[t_idx]}"
            error_counts[error_key] = error_counts.get(error_key, 0) + 1

    return transition_accuracy, error_counts


def compute_epoch_agreement(
    scorer1: np.ndarray,
    scorer2: np.ndarray,
    class_labels: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute inter-rater agreement metrics between two scorers.

    Useful for comparing model predictions against human expert or
    comparing two human scorers.

    Parameters
    ----------
    scorer1, scorer2 : array-like
        Labels from each scorer.

    class_labels : list of str, optional
        Class names for per-class analysis.

    Returns
    -------
    agreement : dict
        Agreement statistics including overall and per-class metrics.
    """
    scorer1 = np.array(scorer1)
    scorer2 = np.array(scorer2)

    if class_labels is None:
        class_labels = sorted(list(set(scorer1) | set(scorer2)))

    agreement = {
        "overall_agreement": float(np.mean(scorer1 == scorer2)),
        "cohen_kappa": float(cohen_kappa_score(scorer1, scorer2)),
        "n_epochs": len(scorer1),
    }

    # Per-class agreement
    for label in class_labels:
        mask1 = scorer1 == label
        mask2 = scorer2 == label

        # When either scorer says this class
        either = mask1 | mask2
        if either.sum() > 0:
            both = mask1 & mask2
            agreement[f"{label}_agreement"] = float(both.sum() / either.sum())
        else:
            agreement[f"{label}_agreement"] = np.nan

    return agreement


def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_format: str = "text",
) -> str:
    """
    Generate a detailed classification report.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.

    y_pred : array-like
        Predicted labels.

    output_format : str
        One of "text" or "dict".

    Returns
    -------
    report : str or dict
        Classification report in requested format.
    """
    if output_format == "dict":
        return classification_report(y_true, y_pred, output_dict=True)
    return classification_report(y_true, y_pred)


def print_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: Optional[List[str]] = None,
) -> str:
    """
    Generate a formatted confusion matrix string.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.

    y_pred : array-like
        Predicted labels.

    class_labels : list of str, optional
        Class names.

    Returns
    -------
    formatted : str
        Formatted confusion matrix.
    """
    if class_labels is None:
        class_labels = sorted(list(set(y_true) | set(y_pred)))

    cm = confusion_matrix(y_true, y_pred, labels=class_labels)

    # Calculate column widths
    max_label_len = max(len(str(label)) for label in class_labels)
    col_width = max(max_label_len, 6)

    lines = ["CONFUSION MATRIX", "=" * 40, ""]

    # Header row
    header = " " * (col_width + 2) + "Predicted"
    lines.append(header)
    header_labels = " " * (col_width + 2) + "".join(
        str(label).rjust(col_width) for label in class_labels
    )
    lines.append(header_labels)
    lines.append("-" * (col_width + 2 + col_width * len(class_labels)))

    # Data rows
    for i, label in enumerate(class_labels):
        prefix = "Actual " if i == len(class_labels) // 2 else "       "
        row = f"{prefix}{str(label).rjust(col_width - 7)} |"
        row += "".join(str(cm[i, j]).rjust(col_width) for j in range(len(class_labels)))
        lines.append(row)

    return "\n".join(lines)
