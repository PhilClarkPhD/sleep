"""
Post-prediction rule-based filtering.

This operates on model *output* (a sequence of predicted states), not on raw
signals, but it lives here because it shares the sleep-scoring domain and has no
backend/model dependencies. Moved verbatim from
backend/app/ml/sleep_functions.py during Phase 1.
"""

from typing import List, Union

import numpy as np


def apply_rule_based_filter(predicted: Union[List, np.ndarray]) -> List:
    """
    Apply rule-based filtering to improve prediction accuracy.

    Handles physiologically implausible transitions like REM directly after Wake.

    Args:
        predicted: List or array of predicted scores ('Wake', 'Non REM', 'REM')

    Returns:
        Filtered list of scores
    """
    if isinstance(predicted, np.ndarray):
        predicted = predicted.tolist()
    else:
        predicted = list(predicted)  # Make a copy to avoid mutation

    # Iterate over the list and apply rules
    for idx in range(2, len(predicted) - 2):
        is_single_epoch = predicted[idx - 2 : idx] == predicted[idx + 1 : idx + 3]
        is_double_epoch = predicted[idx - 2 : idx] == predicted[idx + 2 : idx + 4]
        is_isolated_by_2_and_1 = (
            predicted[idx - 2] == predicted[idx - 1] == predicted[idx + 1]
        )
        is_rem_after_wake = predicted[idx] == "REM" and predicted[idx - 2 : idx] == [
            "Wake",
            "Wake",
        ]

        # Rule 1: If "Wake, Wake, REM", convert "REM" to "Wake"
        if is_rem_after_wake:
            predicted[idx] = "Wake"

        # Rule 2: If the element is isolated by two identical elements on both sides
        if is_single_epoch or is_double_epoch:
            predicted[idx] = predicted[idx - 1]

        # Rule 3: Isolated bout before a legitimate change in sleep state
        elif is_isolated_by_2_and_1:
            predicted[idx] = predicted[idx - 1]

    return predicted
