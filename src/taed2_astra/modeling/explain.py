"""Stage: explain the shipped model with SHAP and check that its reasoning is clinically plausible.

Two outputs, both computed on a sample of the held-out test split:

- Global explainability: mean |SHAP| per clinical variable, split into the part that comes
  from its measured value and the part that comes from its missing-value indicator, plus the
  share of the model's reasoning that comes from *whether* a measurement exists at all.
- Direction checks: for the variables listed in ``params.yaml: explain.expected_direction``,
  the Spearman correlation between the measured value and its SHAP contribution must have
  the clinically expected sign. ``tests/test_explainability.py`` turns these into release gates.

SHAP values are in log-odds (the booster's raw output), not probability points: the model is
trained with balanced class weights, so its probabilities are not calibrated risks.
"""

import json
from dataclasses import dataclass

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.pipeline import Pipeline

from taed2_astra.config import (
    METRICS_DIR,
    SHAP_PATH,
    TEST_PATH,
    get_logger,
    get_tracking_uri,
    load_params,
)
from taed2_astra.features.build_features import split_xy
from taed2_astra.modeling.predict import load_model
from taed2_astra.visualization.plots import save_figure

log = get_logger(__name__)

INDICATOR_PREFIX = "missingindicator_"  # name SimpleImputer(add_indicator=True) gives its indicator columns


@dataclass
class Explanation:
    """SHAP values of one sample, in model-feature space and grouped back to the raw inputs."""

    raw: pd.DataFrame  # the sample as the API would receive it (NaN = not measured)
    values: pd.DataFrame  # one column per model feature (78), log-odds contributions
    grouped: pd.DataFrame  # one column per raw input (41): value + indicator contributions
    base_value: float
    max_additivity_error: float


def input_of(feature: str, inputs: list[str]) -> str:
    """Return the raw input column a model feature was derived from.

    Handles the imputer's missing-value indicators and one-hot columns (``<input>_<category>``),
    so the mapping keeps working if params.yaml ever describes a dataset with categorical columns.
    """
    name = feature.removeprefix(INDICATOR_PREFIX)
    if name in inputs:
        return name
    matches = [col for col in inputs if name.startswith(f"{col}_")]
    if not matches:
        raise KeyError(f"Model feature '{feature}' does not come from any input column")
    return max(matches, key=len)


def explain(model: Pipeline, x: pd.DataFrame) -> Explanation:
    """Return the SHAP values of ``x`` for a fitted Pipeline(preprocessor, tree estimator)."""
    preprocessor, estimator = model.named_steps["preprocessor"], model.named_steps["estimator"]
    transformed = preprocessor.transform(x)
    features = list(preprocessor.get_feature_names_out())
    explainer = shap.TreeExplainer(estimator)
    result = explainer(transformed)

    values = pd.DataFrame(result.values, columns=features, index=x.index)
    groups = [input_of(feature, list(x.columns)) for feature in features]
    grouped = values.T.groupby(groups, sort=False).sum().T.reindex(columns=x.columns, fill_value=0.0)

    # SHAP is additive: base value + contributions must rebuild the booster's raw score.
    base_value = float(np.ravel(result.base_values)[0])
    error = np.abs(values.sum(axis=1).to_numpy() + base_value - estimator.decision_function(transformed)).max()
    return Explanation(x, values, grouped, base_value, float(error))


def missing_indicator_share(values: pd.DataFrame) -> float:
    """Return the fraction of total |SHAP| carried by missing-value indicators."""
    magnitude = values.abs().sum()
    indicators = [name for name in values.columns if name.startswith(INDICATOR_PREFIX)]
    return float(magnitude[indicators].sum() / magnitude.sum())


def importance(exp: Explanation) -> dict[str, dict[str, float]]:
    """Return mean |SHAP| per raw input, split into the value part and the missing-indicator part."""
    table = {}
    for col in exp.raw.columns:
        value_part = float(exp.values[col].abs().mean()) if col in exp.values else 0.0
        indicator = f"{INDICATOR_PREFIX}{col}"
        indicator_part = float(exp.values[indicator].abs().mean()) if indicator in exp.values else 0.0
        # total = value + indicator, so the ranking matches the stacked bars of the importance figure.
        table[col] = {"total": value_part + indicator_part, "value": value_part, "missing_indicator": indicator_part}
    return dict(sorted(table.items(), key=lambda item: item[1]["total"], reverse=True))


def direction_check(raw: pd.Series, contribution: pd.Series, expected: int, settings: dict) -> dict:
    """Return whether higher values of one input push the risk in the clinically expected direction.

    Only rows where the variable was measured count: for a missing value the model sees the
    imputed median, which says nothing about the direction it learned.
    """
    measured = raw.notna()
    n_measured = int(measured.sum())
    check = {"expected": int(expected), "n_measured": n_measured, "spearman": None, "passed": False}
    if n_measured < settings["min_measured_rows"]:
        check["reason"] = f"only {n_measured} measured rows (< {settings['min_measured_rows']})"
        return check
    rho = float(spearmanr(raw[measured], contribution[measured]).statistic)
    check["spearman"] = rho
    check["passed"] = bool(np.sign(rho) == np.sign(expected) and abs(rho) >= settings["min_abs_spearman"])
    if not check["passed"]:
        check["reason"] = (
            f"spearman {rho:+.3f}, expected sign {expected:+d} and |rho| >= {settings['min_abs_spearman']}"
        )
    return check


def direction_checks(exp: Explanation, settings: dict) -> dict[str, dict]:
    """Run ``direction_check`` for every variable in ``explain.expected_direction``."""
    checks = {}
    for col, expected in settings["expected_direction"].items():
        if col not in exp.raw.columns:
            raise KeyError(f"explain.expected_direction names '{col}', which is not a model input")
        checks[col] = direction_check(exp.raw[col], exp.values[col], expected, settings)
    return checks


def plot_importance(table: dict[str, dict[str, float]], top_k: int):
    """Save a horizontal bar chart of the top inputs, split into value and missingness parts."""
    top = list(table.items())[:top_k][::-1]  # largest at the top of the chart
    labels = [name for name, _ in top]
    value_part = [row["value"] for _, row in top]
    indicator_part = [row["missing_indicator"] for _, row in top]
    figure, axis = plt.subplots(figsize=(8, 0.38 * len(top) + 1.4))
    axis.barh(labels, value_part, color="#3A7D78", label="measured value", zorder=3)
    axis.barh(labels, indicator_part, left=value_part, color="#D05A47", label="was it measured?", zorder=3)
    axis.set_xlabel("mean |SHAP| (log-odds)")
    axis.set_title(f"What drives the sepsis alert: top {len(top)} inputs")
    axis.grid(axis="x", color="#D9E1DF", linewidth=0.8, zorder=0)
    axis.legend(loc="lower right", frameon=False)
    figure.tight_layout()
    path = save_figure(figure, "shap_importance")
    plt.close(figure)
    return path


def plot_beeswarm(exp: Explanation, ranking: list[str], top_k: int):
    """Save a SHAP beeswarm of the grouped contributions, coloured by the raw measured value."""
    grouped = shap.Explanation(
        values=exp.grouped.to_numpy(),
        base_values=np.full(len(exp.grouped), exp.base_value),
        data=exp.raw.to_numpy(dtype=float),  # NaN (not measured) is drawn in grey
        feature_names=list(exp.raw.columns),
    )
    columns = list(exp.raw.columns)
    order = np.array([columns.index(name) for name in ranking])  # same order as the importance figure
    shap.plots.beeswarm(grouped, order=order, max_display=top_k, show=False)
    figure = plt.gcf()
    figure.suptitle("SHAP contributions per input (log-odds)", y=1.0)
    path = save_figure(figure, "shap_beeswarm")
    plt.close(figure)
    return path


def build_report(exp: Explanation, name: str, settings: dict) -> dict:
    """Return the JSON-serialisable summary written to metrics/shap.json."""
    checks = direction_checks(exp, settings)
    return {
        "model": name,
        "output": "log-odds",
        "n_rows": len(exp.raw),
        "base_value": exp.base_value,
        "max_additivity_error": exp.max_additivity_error,
        "missing_indicator_share": missing_indicator_share(exp.values),
        "direction_checks_passed": all(check["passed"] for check in checks.values()),
        "direction_checks": checks,
        "importance": importance(exp),
    }


def main() -> None:
    """Explain the shipped model on a test sample and write metrics/shap.json and two figures."""
    params = load_params()
    settings, name = params["explain"], params["train"]["model"]

    x_test, _ = split_xy(pd.read_parquet(TEST_PATH), params["dataset"])
    sample = x_test.sample(n=min(settings["sample_rows"], len(x_test)), random_state=settings["random_state"])
    exp = explain(load_model(), sample)
    report = build_report(exp, name, settings)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(SHAP_PATH, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")  # POSIX text file: keeps pre-commit's end-of-file-fixer from rewriting it
    ranking = list(report["importance"])
    figures = [plot_importance(report["importance"], settings["top_k"]), plot_beeswarm(exp, ranking, settings["top_k"])]

    mlflow.set_tracking_uri(get_tracking_uri(params))
    mlflow.set_experiment(params["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name=f"explain-{name}"):
        mlflow.set_tags({"stage": "explain", "candidate": name, "dataset": params["dataset"]["name"]})
        mlflow.log_params({"n_rows": report["n_rows"], "random_state": settings["random_state"]})
        mlflow.log_metrics(
            {
                "missing_indicator_share": report["missing_indicator_share"],
                "max_additivity_error": report["max_additivity_error"],
                "direction_checks_passed": float(report["direction_checks_passed"]),
            }
        )
        mlflow.log_artifact(str(SHAP_PATH))
        for path in figures:
            mlflow.log_artifact(str(path))

    log.info(
        "Missing-indicator share %.1f%%; direction checks %s; top inputs: %s",
        100 * report["missing_indicator_share"],
        "passed" if report["direction_checks_passed"] else "FAILED",
        ", ".join(list(report["importance"])[:5]),
    )


if __name__ == "__main__":
    main()
