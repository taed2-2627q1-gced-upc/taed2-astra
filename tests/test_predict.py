"""Model contract tests on a small fitted model: what training and serving promise, without data on disk."""

import numpy as np
import pandas as pd
import pytest

from taed2_astra.config import load_params
from taed2_astra.modeling import predict as predict_module
from taed2_astra.modeling.registry import build_model

SHIPPED = load_params()["train"]["model"]


@pytest.fixture(name="toy")
def toy_fixture() -> tuple[pd.DataFrame, pd.Series]:
    """300 rows where the label depends on `a`, with missing values in `b`."""
    rng = np.random.default_rng(1)
    x = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300), "c": rng.normal(size=300)})
    x.loc[rng.random(300) < 0.3, "b"] = np.nan
    y = pd.Series((x["a"] + rng.normal(scale=0.3, size=300) > 0.8).astype(int))
    return x, y


@pytest.fixture(name="served")
def served_fixture(toy, monkeypatch):
    """The shipped candidate fitted on the toy data and swapped in for models/model.pkl."""
    x, y = toy
    model = build_model(SHIPPED, load_params()["train"]).fit(x, y)
    monkeypatch.setattr(predict_module, "load_model", lambda: model)
    return model


def test_training_is_deterministic(toy):
    """Same data and params must give the same model, or no metric in the report can be reproduced."""
    x, y = toy
    params = load_params()["train"]
    first = build_model(SHIPPED, params).fit(x, y).predict_proba(x)
    second = build_model(SHIPPED, params).fit(x, y).predict_proba(x)
    np.testing.assert_array_equal(first, second)


@pytest.mark.usefixtures("served")
def test_prediction_is_one_valid_probability_per_record(toy):
    """Each record gets a probability in [0, 1] and a 0/1 decision consistent with the threshold."""
    x, _ = toy
    threshold = load_params()["evaluate"]["threshold"]
    results = predict_module.predict(x.to_dict(orient="records"))
    assert len(results) == len(x)
    for result in results:
        assert 0.0 <= result["risk_probability"] <= 1.0
        assert result["prediction"] == int(result["risk_probability"] >= threshold)


@pytest.mark.usefixtures("served")
def test_prediction_ignores_key_order(toy):
    """Clients send JSON objects, whose key order is not guaranteed."""
    x, _ = toy
    reordered = x[list(reversed(x.columns))]
    assert predict_module.predict(x.to_dict(orient="records")) == predict_module.predict(
        reordered.to_dict(orient="records")
    )


@pytest.mark.usefixtures("served")
def test_unknown_fields_do_not_change_the_prediction(toy):
    """An extra key (e.g. a patient identifier) must never become a feature at serving time."""
    record = toy[0].iloc[0].to_dict()
    assert predict_module.predict([record]) == predict_module.predict([{**record, "Patient_ID": 123}])


@pytest.mark.usefixtures("served")
def test_missing_measurements_are_imputed_not_rejected():
    """At the bedside most labs are absent; an empty record must still score."""
    results = predict_module.predict([{}, {"a": 1.0}])
    assert len(results) == 2
    assert all(0.0 <= result["risk_probability"] <= 1.0 for result in results)


@pytest.mark.usefixtures("served")
def test_risk_follows_the_signal():
    """The toy label rises with `a`, so a fitted model's risk must rise with it too."""
    low, high = predict_module.predict([{"a": -2.0}, {"a": 3.0}])
    assert high["risk_probability"] > low["risk_probability"]
