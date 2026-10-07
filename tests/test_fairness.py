"""Contract tests for the AIF360 fairness audit and the Reweighing mitigation. They need no data on disk."""

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from taed2_astra.config import load_params
from taed2_astra.modeling.fairness import (
    audit,
    fit_params,
    group_metrics,
    outcomes,
    protected_groups,
    sample_weights,
)
from taed2_astra.modeling.registry import SubsampleClassifier, build_model


def make_params(method: str = "reweighing", attribute: str = "g") -> dict:
    """A params.yaml-shaped dict with one binary and one continuous protected attribute."""
    return {
        "dataset": {"classes": [0, 1]},
        "fairness": {
            "favorable_label": 1,
            "attributes": {"g": {"privileged": 1}, "age": {"cutoff": 65, "privileged": 0}},
            "mitigation": {"method": method, "attribute": attribute},
        },
    }


@pytest.fixture(name="skewed")
def skewed_fixture() -> tuple[pd.DataFrame, pd.Series]:
    """A frame where the label is far more common in the privileged group than in the other."""
    rng = np.random.default_rng(0)
    g = rng.integers(0, 2, size=2000).astype(float)
    x = pd.DataFrame({"g": g, "age": rng.uniform(18, 90, size=2000), "f": rng.normal(size=2000)})
    y = pd.Series((rng.random(2000) < np.where(g == 1, 0.30, 0.05)).astype(int), name="label")
    return x, y


def test_metrics_match_hand_computed_rates():
    """Equal alert rates can hide unequal recall: equal opportunity must expose what parity metrics miss."""
    group = pd.Series([1.0, 1, 1, 1, 0, 0, 0, 0], name="g")
    y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 1, 0, 1, 0])  # privileged: recall 1, no false alarms; other: recall 0.5, FPR 0.5
    metrics = group_metrics(y_true, y_pred, group, privileged=1, labels_pair=(1, 0))
    assert metrics["disparate_impact"] == pytest.approx(1.0)
    assert metrics["statistical_parity_difference"] == pytest.approx(0.0)
    assert metrics["equal_opportunity_difference"] == pytest.approx(-0.5)
    assert metrics["average_odds_difference"] == pytest.approx(0.0)  # a recall loss offset by extra false alarms
    assert metrics["recall_unprivileged"] == pytest.approx(0.5)


def test_rows_missing_the_attribute_belong_to_no_group():
    """AIF360 rejects NaN; a row with an unknown group must be left out, not crash or join a group."""
    group = pd.Series([1.0, 0.0, np.nan, 1.0, 0.0], name="g")
    metrics = group_metrics(np.array([1, 1, 1, 0, 0]), np.array([1, 1, 0, 0, 0]), group, 1, (1, 0))
    assert metrics["n_rows"] == 4


def test_cutoff_binarises_and_keeps_missing_as_missing():
    """A missing age is unknown, not young: binarising must not silently assign it to a group."""
    x = pd.DataFrame({"g": [0, 1, 1], "age": [64.9, 65.0, np.nan]})
    groups = protected_groups(x, make_params()["fairness"]["attributes"])
    assert groups["g"].tolist() == [0.0, 1.0, 1.0]
    assert groups["age"].iloc[:2].tolist() == [0.0, 1.0]
    assert np.isnan(groups["age"].iloc[2])


def test_unfavorable_label_is_the_other_class():
    """The label pair comes from dataset.classes, so no label value is hardcoded in src/."""
    assert outcomes(make_params()) == (1, 0)


def test_audit_reports_every_configured_attribute(skewed):
    """Adding an attribute to params.yaml must be enough to audit it."""
    x, y = skewed
    report = audit(x, y, y.to_numpy(), make_params())
    assert set(report) == {"g", "age"}
    assert report["g"]["equal_opportunity_difference"] == pytest.approx(0.0)  # a perfect model has equal recall


def test_repository_attributes_are_complete_columns():
    """Reweighing needs a weight for every row, so its attribute must be one the data contract keeps complete."""
    params = load_params()
    assert params["fairness"]["mitigation"]["attribute"] in params["validation"]["complete"]
    assert set(params["fairness"]["attributes"]) <= set(params["validation"]["complete"])


def test_no_mitigation_means_unweighted_training(skewed):
    """With method none, training must be byte-for-byte the unmitigated fit: no weights, no extra fit arguments."""
    x, y = skewed
    assert sample_weights(x, y, make_params(method="none")) is None
    assert not fit_params(build_model("logistic_regression", load_params()["train"]), None)


def test_reweighing_makes_the_label_independent_of_the_group(skewed):
    """After Reweighing, the weighted positive rate must be the same in both groups: that is its whole promise."""
    x, y = skewed
    weights = sample_weights(x, y, make_params())
    privileged = (x["g"] == 1).to_numpy()
    rate = [np.average(y[mask], weights=weights[mask]) for mask in (privileged, ~privileged)]
    assert rate[0] == pytest.approx(rate[1])
    assert rate[0] == pytest.approx(y.mean())  # the overall prevalence is preserved


def test_unknown_mitigation_is_rejected(skewed):
    """A typo in params.yaml must fail loudly, not silently train an unmitigated model."""
    x, y = skewed
    with pytest.raises(ValueError, match="reweighting"):
        sample_weights(x, y, make_params(method="reweighting"))


def test_reweighing_refuses_an_incomplete_attribute(skewed):
    """Rows without a group would get no meaningful weight; refusing is safer than guessing one."""
    x, y = skewed
    x.loc[0, "g"] = np.nan
    with pytest.raises(ValueError, match="validation.complete"):
        sample_weights(x, y, make_params())


@pytest.mark.parametrize("candidate", ["logistic_regression", "ensemble_soft", "ensemble_stacking"])
def test_weights_reach_the_estimator(skewed, candidate):
    """The weights must change the fitted model; a routing mistake would make the mitigation a silent no-op.

    Ensembles take **fit_params instead of a named sample_weight, so a signature check alone would reject them.
    """
    x, y = skewed
    weights = sample_weights(x, y, make_params())
    plain = build_model(candidate, load_params()["train"]).fit(x, y)
    weighted = build_model(candidate, load_params()["train"])
    weighted.fit(x, y, **fit_params(weighted, weights))
    assert not np.allclose(plain.predict_proba(x), weighted.predict_proba(x))


def test_estimator_without_sample_weight_is_rejected():
    """TabPFN's wrapper cannot take weights; the benchmark relies on this error to skip it instead of crashing."""
    model = Pipeline([("estimator", SubsampleClassifier(LogisticRegression()))])
    with pytest.raises(ValueError, match="sample_weight"):
        fit_params(model, np.ones(3))


def test_serving_does_not_import_aif360():
    """AIF360 is an offline audit: importing it in the API would add start-up time and log noise to every replica."""
    probe = "import sys, taed2_astra.api.main; print('aif360' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False"
