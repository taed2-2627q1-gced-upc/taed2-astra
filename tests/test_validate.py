"""Data validation tests: each expectation must catch the defect it exists for, on synthetic splits."""

import numpy as np
import pandas as pd
import pytest

from taed2_astra.config import load_params
from taed2_astra.data.validate import build_report, expected_columns

DATASET = {"target": "label", "classes": [0, 1], "group": "patient"}
RULES = {
    "min_rows": 50,
    "label_rate": [0.1, 0.5],
    "binary": ["flag"],
    "complete": ["age"],
    "dtypes": ["int64", "float64"],
    "range_mostly": 0.9,
    "ranges": {"age": [0, 120], "flag": [0, 1], "lactate": [0, 35]},
}


@pytest.fixture(name="split")
def split_fixture() -> pd.DataFrame:
    """A clean split: 100 rows, 20 patients, 20 % positives, some missing lab values."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "age": rng.uniform(18, 90, size=100),
            "flag": rng.integers(0, 2, size=100),
            "lactate": rng.uniform(0.5, 10, size=100),
            "label": np.r_[np.ones(20, dtype=int), np.zeros(80, dtype=int)],
            "patient": np.arange(100) // 5,
        }
    )
    df.loc[::7, "lactate"] = np.nan
    return df


def report_for(df: pd.DataFrame) -> dict:
    """Validate a single split under the synthetic rules."""
    return build_report({"train": df}, DATASET, RULES)


def test_clean_split_passes_without_warnings(split):
    """The baseline: a split that honours every rule raises nothing."""
    report = report_for(split)
    assert report["success"], report["critical_failures"]
    assert not report["warnings"]


def test_missing_feature_column_is_critical(split):
    """A dropped column changes the model's input, so the pipeline must stop."""
    report = report_for(split.drop(columns="lactate"))
    assert not report["success"]
    assert "train: expect_table_columns_to_match_set" in report["critical_failures"]


def test_unexpected_extra_column_is_critical(split):
    """An extra column would silently become a new feature."""
    report = report_for(split.assign(leaky_outcome=1))
    assert not report["success"]


def test_text_typed_feature_is_critical(split):
    """A numeric column read as text would be one-hot encoded instead of scaled."""
    report = report_for(split.assign(age=split["age"].astype(str)))
    assert "train: expect_column_values_to_be_in_type_list(age)" in report["critical_failures"]


def test_unknown_label_value_is_critical(split):
    """A label outside the declared classes means the target was re-coded upstream."""
    split.loc[0, "label"] = 2
    report = report_for(split)
    assert "train: expect_column_distinct_values_to_be_in_set(label)" in report["critical_failures"]


def test_missing_label_is_critical(split):
    """Rows without a label cannot be trained or scored."""
    split["label"] = split["label"].astype(float)
    split.loc[:2, "label"] = np.nan
    report = report_for(split)
    assert "train: expect_column_values_to_not_be_null(label)" in report["critical_failures"]


def test_implausible_prevalence_is_critical(split):
    """A split with no positives (e.g. a label join that failed) must not reach training."""
    report = report_for(split.assign(label=0))
    assert "train: expect_column_mean_to_be_between(label)" in report["critical_failures"]


def test_gap_in_a_complete_column_is_critical(split):
    """Columns charted for every row must stay complete."""
    split.loc[3, "age"] = np.nan
    report = report_for(split)
    assert "train: expect_column_values_to_not_be_null(age)" in report["critical_failures"]


def test_non_binary_flag_is_critical(split):
    """A binary indicator with a third value means the encoding changed."""
    split.loc[0, "flag"] = 3
    report = report_for(split)
    assert "train: expect_column_values_to_be_in_set(flag)" in report["critical_failures"]


def test_few_out_of_range_values_are_tolerated(split):
    """Below the `range_mostly` budget, a charting error does not block the pipeline."""
    split.loc[:4, "lactate"] = 500.0
    report = report_for(split)
    assert report["success"]
    assert not report["warnings"]


def test_systematic_out_of_range_values_warn_without_blocking(split):
    """A unit change across a column is flagged for review, not silently trained on."""
    split["lactate"] = split["lactate"] * 1000
    report = report_for(split)
    assert report["success"]
    assert report["warnings"] == ["train: expect_column_values_to_be_between(lactate)"]


def test_patient_in_both_splits_is_critical(split):
    """A shared patient leaks their own trajectory into the test set."""
    test = split.copy()
    report = build_report({"train": split, "test": test}, DATASET, RULES)
    assert report["group_overlap"] == 20
    assert not report["success"]


def test_disjoint_splits_pass(split):
    """The grouped split the prepare stage produces must pass the overlap check."""
    test = split.assign(patient=split["patient"] + 1000)
    report = build_report({"train": split, "test": test}, DATASET, RULES)
    assert report["success"], report["critical_failures"]
    assert report["group_overlap"] == 0


def test_params_validation_rules_are_coherent():
    """The real rules in params.yaml must be satisfiable before they can be trusted."""
    params = load_params()
    dataset, rules = params["dataset"], params["validation"]
    for column, (low, high) in rules["ranges"].items():
        assert low < high, f"Empty range for {column}"
    assert set(rules["complete"]) <= set(rules["ranges"])
    assert set(rules["binary"]) <= set(rules["ranges"])
    assert 0 < rules["label_rate"][0] < rules["label_rate"][1] < 1
    assert dataset["target"] in expected_columns(dataset, rules)
