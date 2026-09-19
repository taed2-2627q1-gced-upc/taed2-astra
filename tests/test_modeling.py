"""Contract tests for the registry, features and metrics. They need no data on disk."""

import numpy as np
import pandas as pd
import pytest

from taed2_astra.config import load_params
from taed2_astra.features.build_features import split_xy
from taed2_astra.modeling.evaluate import compute_metrics
from taed2_astra.modeling.registry import build_model

CANDIDATES = sorted(load_params()["train"]["candidates"])


@pytest.fixture(name="toy")
def toy_fixture() -> tuple[pd.DataFrame, pd.Series]:
    """A small frame with missing values and an imbalanced binary label."""
    rng = np.random.default_rng(0)
    x = pd.DataFrame(
        {
            "a": rng.normal(size=400),
            "b": rng.normal(size=400),
            "c": rng.integers(0, 2, size=400).astype(float),
        }
    )
    x.loc[rng.random(400) < 0.3, "b"] = np.nan
    y = pd.Series((x["a"] + rng.normal(scale=0.5, size=400) > 1.0).astype(int), name="label")
    return x, y


@pytest.mark.parametrize("name", CANDIDATES)
def test_candidate_fits_with_missing_values(name, toy):
    """Every candidate in params.yaml must fit on NaN-bearing data and emit one probability per row."""
    try:
        model = build_model(name, load_params()["train"])
    except ImportError as error:
        pytest.skip(str(error))
    x, y = toy
    proba = model.fit(x, y).predict_proba(x)[:, 1]
    assert proba.shape == (len(x),)
    assert np.all((proba >= 0.0) & (proba <= 1.0))
    assert list(model.feature_names_in_) == list(x.columns)


def test_unknown_candidate_is_rejected():
    """A typo in params.yaml must fail loudly, not fall back to a default model."""
    params = {"random_state": 0, "candidates": {"bad": {"estimator": "does_not_exist"}}}
    with pytest.raises(KeyError, match="does_not_exist"):
        build_model("bad", params)


def test_split_xy_never_leaks_label_or_group():
    """The label and the grouping column are excluded from the feature matrix."""
    df = pd.DataFrame({"f": [1.0, 2.0], "label": [0, 1], "pid": [7, 7]})
    x, y = split_xy(df, {"target": "label", "group": "pid"})
    assert list(x.columns) == ["f"]
    assert y.tolist() == [0, 1]


def test_compute_metrics_is_flat_and_numeric():
    """Metrics must be a flat name -> float mapping so MLflow and DVC can log them."""
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.4, 0.6, 0.9])
    metrics = compute_metrics(y, (proba >= 0.5).astype(int), proba)
    assert metrics["roc_auc"] == 1.0
    assert metrics["f1"] == 1.0
    assert all(isinstance(value, float) for value in metrics.values())
