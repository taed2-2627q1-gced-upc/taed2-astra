"""Stage 5: audit the shipped model's decisions for group fairness, and the Reweighing mitigation.

AIF360 rejects missing values and most lab columns are mostly missing, so only the
label, the decision and one protected attribute are ever converted to an AIF360
dataset. The model and its preprocessing never see an AIF360 object, and neither
does the API: serving does not import this module.
"""

import inspect

import mlflow
import numpy as np
import pandas as pd
from aif360.algorithms.preprocessing import Reweighing
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import has_fit_parameter

from taed2_astra.config import (
    FAIRNESS_PATH,
    METRICS_DIR,
    TEST_PATH,
    get_logger,
    get_tracking_uri,
    load_params,
    write_json,
)
from taed2_astra.features.build_features import split_xy
from taed2_astra.modeling.predict import load_model

log = get_logger(__name__)

_LABEL = "label"  # column name inside the AIF360 frame only; never looked up in the data


def outcomes(params: dict) -> tuple[int, int]:
    """Return the (favorable, unfavorable) label pair, the latter being the other class in dataset.classes."""
    favorable = params["fairness"]["favorable_label"]
    (unfavorable,) = (label for label in params["dataset"]["classes"] if label != favorable)
    return favorable, unfavorable


def protected_groups(x: pd.DataFrame, attributes: dict) -> pd.DataFrame:
    """Return one numeric group column per attribute, binarised at its cutoff when it has one; NaN stays NaN."""
    groups = {}
    for name, spec in attributes.items():
        column = x[name].astype(float)
        if "cutoff" in spec:
            column = (column >= spec["cutoff"]).astype(float).where(column.notna())
        groups[name] = column
    return pd.DataFrame(groups, index=x.index)


def _conditions(group: pd.Series, privileged: float) -> tuple[list[dict], list[dict]]:
    """Return AIF360's (unprivileged, privileged) group conditions: every other observed value is unprivileged."""
    others = sorted(value for value in group.dropna().unique() if value != privileged)
    return [{group.name: value} for value in others], [{group.name: privileged}]


def to_dataset(labels: pd.Series | np.ndarray, group: pd.Series, labels_pair: tuple[int, int]) -> BinaryLabelDataset:
    """Return an AIF360 dataset holding only a label column and one complete protected attribute."""
    favorable, unfavorable = labels_pair
    frame = pd.DataFrame({_LABEL: np.asarray(labels, dtype=float), group.name: group.to_numpy(dtype=float)})
    return BinaryLabelDataset(
        df=frame,
        label_names=[_LABEL],
        protected_attribute_names=[group.name],
        favorable_label=favorable,
        unfavorable_label=unfavorable,
    )


def group_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    group: pd.Series,
    privileged: float,
    labels_pair: tuple[int, int],
) -> dict[str, float]:
    """Return AIF360 group fairness metrics for one attribute, as unprivileged minus privileged.

    Rows where the attribute is missing are left out: they belong to neither group.
    """
    known = group.notna().to_numpy()
    truth = to_dataset(np.asarray(y_true)[known], group[known], labels_pair)
    decided = truth.copy()
    decided.labels = np.asarray(y_pred, dtype=float)[known].reshape(-1, 1)
    unprivileged, privileged_groups = _conditions(group, privileged)
    metric = ClassificationMetric(truth, decided, unprivileged_groups=unprivileged, privileged_groups=privileged_groups)
    return {
        "n_rows": int(known.sum()),
        "base_rate_privileged": float(metric.base_rate(privileged=True)),
        "base_rate_unprivileged": float(metric.base_rate(privileged=False)),
        "recall_privileged": float(metric.true_positive_rate(privileged=True)),
        "recall_unprivileged": float(metric.true_positive_rate(privileged=False)),
        "disparate_impact": float(metric.disparate_impact()),
        "statistical_parity_difference": float(metric.statistical_parity_difference()),
        "equal_opportunity_difference": float(metric.equal_opportunity_difference()),
        "average_odds_difference": float(metric.average_odds_difference()),
    }


def audit(x: pd.DataFrame, y_true: pd.Series, y_pred: np.ndarray, params: dict) -> dict[str, dict[str, float]]:
    """Return group_metrics for every attribute in fairness.attributes, keyed by attribute."""
    attributes = params["fairness"]["attributes"]
    groups = protected_groups(x, attributes)
    labels_pair = outcomes(params)
    return {
        name: group_metrics(y_true, y_pred, groups[name], spec["privileged"], labels_pair)
        for name, spec in attributes.items()
    }


def sample_weights(x: pd.DataFrame, y: pd.Series, params: dict) -> np.ndarray | None:
    """Return Reweighing training weights, or None when fairness.mitigation.method is none.

    Reweighing gives each (group, label) cell the weight that makes group and label
    independent in the weighted training set, without editing a single feature value.
    """
    fairness = params["fairness"]
    method = fairness["mitigation"]["method"]
    if method == "none":
        return None
    if method != "reweighing":
        raise ValueError(f"Unknown fairness.mitigation.method '{method}'. Known: none, reweighing")
    name = fairness["mitigation"]["attribute"]
    spec = fairness["attributes"][name]
    group = protected_groups(x, {name: spec})[name]
    if group.isna().any():
        raise ValueError(f"Reweighing needs {name} on every row; add it to validation.complete.")
    unprivileged, privileged = _conditions(group, spec["privileged"])
    reweighed = Reweighing(unprivileged_groups=unprivileged, privileged_groups=privileged).fit_transform(
        to_dataset(y, group, outcomes(params))
    )
    return reweighed.instance_weights


def takes_sample_weight(estimator: BaseEstimator) -> bool:
    """Return whether fit() can honour sample_weight, members included for ensembles.

    VotingClassifier and StackingClassifier take ``**fit_params`` rather than a named
    sample_weight, and pass the weights on to every member, so each member must take them too.
    """
    members = [member for _, member in getattr(estimator, "estimators", [])]
    forwards = any(p.kind is p.VAR_KEYWORD for p in inspect.signature(estimator.fit).parameters.values())
    accepts = has_fit_parameter(estimator, "sample_weight") or (bool(members) and forwards)
    return accepts and all(takes_sample_weight(member) for member in members)


def fit_params(model: Pipeline, weights: np.ndarray | None) -> dict:
    """Return the fit() keyword arguments that route training weights to the pipeline's estimator."""
    if weights is None:
        return {}
    estimator = model.named_steps["estimator"]
    if not takes_sample_weight(estimator):
        raise ValueError(f"{type(estimator).__name__} takes no sample_weight, so Reweighing cannot train it.")
    return {"estimator__sample_weight": weights}


def main() -> None:
    """Audit the shipped model on the test split and write metrics/fairness.json for DVC to track."""
    params = load_params()
    name = params["train"]["model"]

    x_test, y_test = split_xy(pd.read_parquet(TEST_PATH), params["dataset"])
    proba = load_model().predict_proba(x_test)[:, 1]
    report = audit(x_test, y_test, (proba >= params["evaluate"]["threshold"]).astype(int), params)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    write_json(FAIRNESS_PATH, report)

    mlflow.set_tracking_uri(get_tracking_uri(params))
    mlflow.set_experiment(params["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name=f"fairness-{name}"):
        mlflow.set_tags({"stage": "fairness", "candidate": name, "dataset": params["dataset"]["name"]})
        mlflow.log_params({"mitigation": params["fairness"]["mitigation"]["method"], "attributes": list(report)})
        mlflow.log_metrics({f"{attr}_{key}": value for attr, row in report.items() for key, value in row.items()})

    for attr, row in report.items():
        log.info(
            "%s: equal opportunity %+.4f, average odds %+.4f, disparate impact %.3f",
            attr,
            row["equal_opportunity_difference"],
            row["average_odds_difference"],
            row["disparate_impact"],
        )


if __name__ == "__main__":
    main()
