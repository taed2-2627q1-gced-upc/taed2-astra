"""Release gates for the shipped model, scored on the held-out test split.

These need `dvc repro` (or `dvc pull`) to have produced models/model.pkl and the
test split; they skip otherwise, so the fast suite still runs on a fresh clone.
Thresholds live in params.yaml (`model_quality`), next to the parameters they judge.
"""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, recall_score, roc_auc_score

from taed2_astra.config import FAIRNESS_PATH, METRICS_PATH, MODEL_PATH, TEST_PATH, load_params
from taed2_astra.features.build_features import split_xy
from taed2_astra.modeling.fairness import audit
from taed2_astra.modeling.predict import load_model

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (MODEL_PATH.exists() and TEST_PATH.exists()),
        reason="Needs models/model.pkl and the test split: run `dvc repro evaluate` or `dvc pull`.",
    ),
]

PARAMS = load_params()
GATES = PARAMS["model_quality"]
SAMPLE_ROWS = 20_000  # perturbation tests: enough rows for a stable mean, fast enough for every run


@pytest.fixture(name="scored", scope="module")
def scored_fixture() -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """The test split's features, labels and the shipped model's probabilities."""
    x, y = split_xy(pd.read_parquet(TEST_PATH), PARAMS["dataset"])
    return x, y, load_model().predict_proba(x)[:, 1]


def test_roc_auc_meets_the_release_gate(scored):
    """Ranking quality must not regress below the agreed floor."""
    _, y, proba = scored
    assert roc_auc_score(y, proba) >= GATES["min_roc_auc"]


def test_pr_auc_beats_the_no_skill_baseline_by_the_agreed_margin(scored):
    """At 1.7 % prevalence PR-AUC is the honest metric; its no-skill value is the positive rate."""
    _, y, proba = scored
    lift = average_precision_score(y, proba) / y.mean()
    assert lift >= GATES["min_pr_auc_lift"], f"PR-AUC is only {lift:.1f}x the positive rate"


def test_recall_at_the_served_threshold(scored):
    """The API decides at evaluate.threshold; a missed septic patient is the costly error."""
    _, y, proba = scored
    decision = (proba >= PARAMS["evaluate"]["threshold"]).astype(int)
    assert recall_score(y, decision) >= GATES["min_recall"]


@pytest.mark.parametrize("column", GATES["slices"])
def test_no_subgroup_is_served_much_worse(scored, column):
    """Every observed value of a slice column must get similar ranking quality (a basic fairness check)."""
    x, y, proba = scored
    scores = {
        value: roc_auc_score(y[mask], proba[mask])
        for value in x[column].dropna().unique()
        if (mask := (x[column] == value).to_numpy()).any() and y[mask].nunique() == 2
    }
    assert len(scores) >= 2, f"Not enough labelled groups in {column} to compare"
    gap = max(scores.values()) - min(scores.values())
    assert gap <= GATES["max_slice_roc_auc_gap"], f"ROC-AUC by {column}: {scores}"


@pytest.fixture(name="fairness", scope="module")
def fairness_fixture(scored) -> dict[str, dict[str, float]]:
    """AIF360 group metrics of the shipped model's decisions, per protected attribute."""
    x, y, proba = scored
    return audit(x, y, (proba >= PARAMS["evaluate"]["threshold"]).astype(int), PARAMS)


@pytest.mark.parametrize("attribute", PARAMS["fairness"]["attributes"])
def test_missed_sepsis_does_not_concentrate_in_one_group(fairness, attribute):
    """Equal opportunity: a septic patient's chance of being flagged must not depend on sex or age group."""
    gap = fairness[attribute]["equal_opportunity_difference"]
    assert abs(gap) <= GATES["max_equal_opportunity_gap"], f"Recall gap by {attribute}: {gap:+.4f}"


@pytest.mark.parametrize("attribute", PARAMS["fairness"]["attributes"])
def test_error_rates_are_balanced_across_groups(fairness, attribute):
    """Average odds: recall and false-alarm rate together, so one group is not traded off against another."""
    gap = fairness[attribute]["average_odds_difference"]
    assert abs(gap) <= GATES["max_average_odds_gap"], f"Average odds gap by {attribute}: {gap:+.4f}"


def test_fairness_file_matches_the_model_on_disk(fairness):
    """metrics/fairness.json must describe this model, or the model card is quoting a stale audit."""
    with open(FAIRNESS_PATH, encoding="utf-8") as handle:
        recorded = json.load(handle)
    assert recorded.keys() == fairness.keys()
    for attribute, metrics in fairness.items():
        assert recorded[attribute] == pytest.approx(metrics, abs=1e-9)


@pytest.mark.parametrize(("column", "shift"), GATES["directional"].items())
def test_clinical_deterioration_raises_average_risk(scored, column, shift):
    """Worse vitals or labs must not make the model calmer: a directional expectation test."""
    x, _, _ = scored
    sample = x[x[column].notna()].sample(min(SAMPLE_ROWS, int(x[column].notna().sum())), random_state=0)
    worse = sample.assign(**{column: sample[column] + shift})
    model = load_model()
    delta = model.predict_proba(worse)[:, 1].mean() - model.predict_proba(sample)[:, 1].mean()
    assert delta > 0, f"Shifting {column} by {shift} changed mean risk by {delta:+.4f}"


def test_identifiers_are_not_features():
    """Patient IDs carry no clinical meaning; a model reading them would memorise patients."""
    features = set(load_model().feature_names_in_)
    assert PARAMS["dataset"]["target"] not in features
    assert PARAMS["dataset"]["group"] not in features


def test_predictions_are_deterministic(scored):
    """Scoring the same rows twice must give identical probabilities (reproducible reports and API)."""
    x, _, proba = scored
    np.testing.assert_array_equal(load_model().predict_proba(x.head(1000))[:, 1], proba[:1000])


def test_metrics_file_matches_the_model_on_disk(scored):
    """metrics/metrics.json must describe this model, or the model card is quoting a stale run."""
    _, y, proba = scored
    with open(METRICS_PATH, encoding="utf-8") as handle:
        recorded = json.load(handle)
    assert roc_auc_score(y, proba) == pytest.approx(recorded["roc_auc"], abs=1e-6)
