from data_processing.sleep_functions import apply_rule_based_filter
import pytest
import pandas as pd


@pytest.mark.parametrize(
    "scores, expected",
    [
        (  # Single mis-scored epoch
            pd.Series(["Wake", "Wake", "Wake", "Non REM", "Wake", "Wake", "Wake"]),
            ["Wake", "Wake", "Wake", "Wake", "Wake", "Wake", "Wake"],
        ),
        (  # Rem after Wake
            pd.Series(["Wake", "Wake", "Wake", "Wake", "REM", "REM", "REM", "REM"]),
            ["Wake", "Wake", "Wake", "Wake", "Wake", "Wake", "REM", "REM"],
        ),
        (  # Legitimate REM bout
            pd.Series(
                ["Non REM", "Non REM", "REM", "REM", "REM", "Non REM", "Non REM"]
            ),
            ["Non REM", "Non REM", "REM", "REM", "REM", "Non REM", "Non REM"],
        ),
        (  # Illegitimate Non REM bout
            pd.Series(["Wake", "Wake", "Non REM", "Non REM", "Wake", "Wake", "Wake"]),
            ["Wake", "Wake", "Wake", "Wake", "Wake", "Wake", "Wake"],
        ),
        (  # Single mis-scored Non REM near end of Wake bout
            pd.Series(
                [
                    "Wake",
                    "Wake",
                    "Wake",
                    "Non REM",
                    "Wake",
                    "Non REM",
                    "Non REM",
                    "Non REM",
                ]
            ),
            [
                "Wake",
                "Wake",
                "Wake",
                "Wake",
                "Wake",
                "Non REM",
                "Non REM",
                "Non REM",
            ],
        ),
        (
            pd.Series(
                [
                    "Wake",
                    "Wake",
                    "Wake",
                    "Non REM",
                    "Non REM",
                    "Non REM",
                    "Wake",
                    "Wake",
                    "Non REM",
                    "Wake",
                    "Non REM",
                    "Non REM",
                    "Non REM",
                    "Non REM",
                ]
            ),
            [
                "Wake",
                "Wake",
                "Wake",
                "Non REM",
                "Non REM",
                "Non REM",
                "Wake",
                "Wake",
                "Wake",
                "Wake",
                "Non REM",
                "Non REM",
                "Non REM",
                "Non REM",
            ],
        ),
    ],
)
def test_apply_rule_based_filter(scores, expected):
    filtered_scores = apply_rule_based_filter(scores)
    assert filtered_scores == expected
