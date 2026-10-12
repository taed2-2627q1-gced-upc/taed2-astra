"""SHAP explainability: unit tests on synthetic data, then release gates on the shipped model.

The unit tests need no data on disk. The gates are `integration` tests, like those in
test_model_quality.py: they explain models/model.pkl live and skip on a fresh clone.
"""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from taed2_astra.config import MODEL_PATH, SHAP_PATH, TEST_PATH, load_params
from taed2_astra.features.build_features import build_preprocessor
from taed2_astra.modeling import explain as explain_module
from taed2_astra.modeling.explain import (
    build_report,
    direction_check,
    explain,
    input_of,
    missing_indicator_share,
    plot_beeswarm,
    plot_importance,
    sample_test_split,
)
from taed2_astra.modeling.predict import load_model
from taed2_astra.modeling.promote import file_md5

SETTINGS = load_params()["explain"]


@pytest.fixture(name="fitted", scope="module")
def fitted_fixture() -> tuple[Pipeline, pd.DataFrame]:
    """A small booster where risk rises with ``up``, falls with ``down`` and ``noise`` is irrelevant."""
    rng = np.random.default_rng(0)
    n = 1500
    x = pd.DataFrame({"up": rng.normal(size=n), "down": rng.normal(size=n), "noise": rng.normal(size=n)})
    y = (x["up"] - x["down"] + rng.normal(scale=0.5, size=n) > 1.0).astype(int)
    x.loc[rng.random(n) < 0.3, "down"] = np.nan  # missing values, so the imputer adds an indicator
    model = Pipeline(
        [("preprocessor", build_preprocessor()), ("estimator", HistGradientBoostingClassifier(random_state=0))]
    )
    return model.fit(x, y), x


def test_input_of_maps_indicators_and_one_hot_columns():
    """Every model feature is traced back to the raw input it came from."""
    inputs = ["HR", "Unit", "Unit1"]
    assert input_of("HR", inputs) == "HR"
    assert input_of("missingindicator_HR", inputs) == "HR"
    assert input_of("Unit1_yes", inputs) == "Unit1"  # longest prefix wins over "Unit"


def test_input_of_rejects_an_unknown_feature():
    """A feature that maps to no input means the grouping is wrong, so it must fail loudly."""
    with pytest.raises(KeyError, match="Temp"):
        input_of("Temp", ["HR"])


def test_explanation_is_additive_and_grouped_per_input(fitted):
    """Base value + contributions rebuild the raw score, and grouping keeps every contribution."""
    model, x = fitted
    exp = explain(model, x.head(300))
    assert exp.max_additivity_error < 1e-6
    assert list(exp.grouped.columns) == list(x.columns)
    assert "missingindicator_down" in exp.values.columns
    np.testing.assert_allclose(exp.grouped.sum(axis=1), exp.values.sum(axis=1), atol=1e-9)
    assert 0.0 < missing_indicator_share(exp.values) < 1.0


def test_direction_check_accepts_the_true_direction_and_rejects_the_reverse(fitted):
    """Testing the test: the check passes for the learned sign and fails for the opposite one."""
    model, x = fitted
    exp = explain(model, x.head(500))
    settings = {"min_measured_rows": 50, "min_abs_spearman": 0.3}
    assert direction_check(exp.raw["up"], exp.values["up"], 1, settings)["passed"]
    assert direction_check(exp.raw["down"], exp.values["down"], -1, settings)["passed"]
    reversed_check = direction_check(exp.raw["up"], exp.values["up"], -1, settings)
    assert not reversed_check["passed"]
    assert "expected sign" in reversed_check["reason"]
    assert not direction_check(exp.raw["noise"], exp.values["noise"], 1, settings)["passed"]


def test_direction_check_needs_enough_measured_rows(fitted):
    """With too few measured values the check fails instead of passing on noise."""
    model, x = fitted
    exp = explain(model, x.head(500))
    check = direction_check(exp.raw["up"], exp.values["up"], 1, {"min_measured_rows": 10_000, "min_abs_spearman": 0})
    assert not check["passed"]
    assert "measured rows" in check["reason"]


def test_report_ranks_inputs_and_figures_are_written(fitted, tmp_path, monkeypatch):
    """The report is JSON-ready, ranked by total |SHAP|, and both figures land in reports/figures."""
    monkeypatch.setattr(explain_module, "save_figure", lambda fig, name: _save(fig, tmp_path / f"{name}.png"))
    model, x = fitted
    exp = explain(model, x.head(300))
    settings = {"min_measured_rows": 50, "min_abs_spearman": 0.3, "expected_direction": {"up": 1, "down": -1}}
    report = build_report(exp, "toy", settings)
    json.dumps(report)  # must serialise as-is: no numpy types
    totals = [row["total"] for row in report["importance"].values()]
    assert totals == sorted(totals, reverse=True)
    assert report["direction_checks_passed"]
    ranking = list(report["importance"])
    paths = [plot_importance(report["importance"], top_k=2), plot_beeswarm(exp, ranking, top_k=2)]
    assert all(path.stat().st_size > 0 for path in paths)


def test_unknown_variable_in_expected_direction_fails_loudly(fitted):
    """A typo in params.yaml must not silently skip a clinical gate."""
    model, x = fitted
    settings = {"min_measured_rows": 1, "min_abs_spearman": 0, "expected_direction": {"Lactat": 1}}
    with pytest.raises(KeyError, match="Lactat"):
        build_report(explain(model, x.head(100)), "toy", settings)


def _save(figure, path):
    figure.savefig(path)
    return path


# --- Release gates: the shipped model, explained live on the test split (`make promote`, CI) ---------

needs_model = pytest.mark.skipif(
    not (MODEL_PATH.exists() and TEST_PATH.exists()),
    reason="Needs models/model.pkl and the test split: run `dvc repro explain` or `dvc pull`.",
)


@pytest.fixture(name="live_report", scope="module")
def live_report_fixture() -> dict:
    """The SHAP report of models/model.pkl, recomputed now on the same sample the stage uses."""
    params = load_params()
    exp = explain(load_model(), sample_test_split(params))
    return build_report(exp, params["train"]["model"], params["explain"], file_md5(MODEL_PATH))


@pytest.mark.parametrize("variable", sorted(SETTINGS["expected_direction"]))
@pytest.mark.integration
@needs_model
def test_clinical_direction(live_report, variable):
    """Worse values of this variable must push the model towards an alert, not away from it."""
    check = live_report["direction_checks"][variable]
    assert check["passed"], f"{variable}: {check.get('reason')}"


@pytest.mark.integration
@needs_model
def test_explanation_adds_up_to_the_model_output(live_report):
    """If base value + contributions do not rebuild the score, the explanation describes another model."""
    assert live_report["max_additivity_error"] < 1e-6


@pytest.mark.integration
@needs_model
def test_shap_file_matches_the_model_on_disk(live_report):
    """metrics/shap.json must describe this model, or the model card is quoting a stale explanation."""
    with open(SHAP_PATH, encoding="utf-8") as handle:
        recorded = json.load(handle)
    assert recorded["model_md5"] == live_report["model_md5"]
    assert recorded["direction_checks"].keys() == live_report["direction_checks"].keys()
    for name, row in live_report["importance"].items():
        assert recorded["importance"][name] == pytest.approx(row, abs=1e-9)
