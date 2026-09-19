"""Stage: score every candidate in params.yaml on identical cross-validation folds.

One MLflow parent run holds the matrix; one nested run per candidate holds its
hyperparameters, fold metrics, latency, size, emissions and fitted artifact, so
the tracking UI can sort candidates by any column. The test split is never read.
"""

import json
import pickle
import statistics
import time

import mlflow
import numpy as np
import pandas as pd
from codecarbon import EmissionsTracker
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from taed2_astra.config import (
    BENCHMARK_PATH,
    EMISSIONS_DIR,
    METRICS_DIR,
    TRAIN_PATH,
    get_logger,
    get_tracking_uri,
    load_params,
)
from taed2_astra.features.build_features import split_xy
from taed2_astra.modeling.evaluate import compute_metrics
from taed2_astra.modeling.registry import build_model, is_ensemble

log = get_logger(__name__)

Fold = tuple[np.ndarray, np.ndarray]


def make_folds(y: pd.Series, groups: pd.Series | None, n_splits: int, random_state: int) -> list[Fold]:
    """Return the (train_idx, val_idx) pairs every candidate is scored on."""
    if groups is None:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    else:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(splitter.split(np.zeros(len(y)), y, groups))


def cross_validate_candidate(
    name: str, params: dict, x: pd.DataFrame, y: pd.Series, folds: list[Fold]
) -> tuple[dict[str, float], BaseEstimator]:
    """Fit and score one candidate on every fold; return mean/std metrics and the last fitted model."""
    threshold = params["evaluate"]["threshold"]
    per_fold: list[dict[str, float]] = []
    fit_times: list[float] = []
    model = build_model(name, params["train"])
    for train_idx, val_idx in folds:
        model = build_model(name, params["train"])
        start = time.perf_counter()
        model.fit(x.iloc[train_idx], y.iloc[train_idx])
        fit_times.append(time.perf_counter() - start)
        proba = model.predict_proba(x.iloc[val_idx])[:, 1]
        per_fold.append(compute_metrics(y.iloc[val_idx], (proba >= threshold).astype(int), proba))

    summary: dict[str, float] = {}
    for metric in per_fold[0]:
        values = [fold[metric] for fold in per_fold]
        summary[f"{metric}_mean"] = float(np.mean(values))
        summary[f"{metric}_std"] = float(np.std(values))
    summary["fit_time_s_mean"] = float(np.mean(fit_times))
    return summary, model


def measure_latency(model: BaseEstimator, x: pd.DataFrame, batch_size: int, repeats: int = 5) -> float:
    """Return the median predict_proba wall time in milliseconds per 1000 rows."""
    batch = x.iloc[:batch_size]
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        model.predict_proba(batch)
        timings.append(time.perf_counter() - start)
    return statistics.median(timings) * 1000.0 / len(batch) * 1000.0


def add_ensemble_deltas(matrix: pd.DataFrame, rank_by: str) -> pd.DataFrame:
    """Append each candidate's metric gain and latency ratio against the best single model."""
    singles = matrix[~matrix["is_ensemble"]]
    if singles.empty:
        return matrix
    best = singles.loc[singles[f"{rank_by}_mean"].idxmax()]
    matrix[f"gain_{rank_by}_vs_best_single"] = matrix[f"{rank_by}_mean"] - best[f"{rank_by}_mean"]
    matrix["latency_ratio_vs_best_single"] = matrix["latency_ms_per_1k_rows"] / best["latency_ms_per_1k_rows"]
    matrix["best_single"] = best["candidate"]
    return matrix


def _load_training_frame(params: dict) -> pd.DataFrame:
    """Return the training split, optionally capped to benchmark.max_rows for fast iteration."""
    train = pd.read_parquet(TRAIN_PATH)
    max_rows = params["benchmark"].get("max_rows")
    if max_rows and len(train) > max_rows:
        train = train.sample(n=max_rows, random_state=params["train"]["random_state"]).reset_index(drop=True)
        log.warning("benchmark.max_rows=%d is set: results are indicative, not final", max_rows)
    return train


def _run_candidate(name: str, params: dict, x: pd.DataFrame, y: pd.Series, folds: list[Fold]) -> dict | None:
    """Benchmark one candidate inside a nested MLflow run; return its matrix row or None if skipped."""
    spec = params["train"]["candidates"][name]
    bench = params["benchmark"]
    mlflow.set_tags({"candidate": name, "estimator": spec["estimator"], "dataset": params["dataset"]["name"]})
    mlflow.log_params({"random_state": params["train"]["random_state"], **spec})
    try:
        build_model(name, params["train"])  # fail fast on missing optional dependencies
    except ImportError as error:
        log.warning("Skipping %s: %s", name, error)
        mlflow.set_tags({"status": "skipped", "reason": str(error)})
        return None

    tracker = EmissionsTracker(project_name=name, output_dir=str(EMISSIONS_DIR), log_level="error")
    tracker.start()
    try:
        summary, model = cross_validate_candidate(name, params, x, y, folds)
    finally:
        emissions = tracker.stop() or 0.0

    summary["emissions_kg_co2"] = float(emissions)
    summary["latency_ms_per_1k_rows"] = measure_latency(model, x, bench["latency_batch_size"])
    summary["model_size_mb"] = len(pickle.dumps(model)) / 1e6
    mlflow.log_metrics(summary)
    mlflow.sklearn.log_model(model, name="model")
    mlflow.set_tag("status", "completed")
    log.info("%s: %s", name, {key: round(value, 4) for key, value in summary.items()})
    return {
        "candidate": name,
        "estimator": spec["estimator"],
        "is_ensemble": is_ensemble(name, params["train"]),
        **summary,
    }


def main() -> None:
    """Benchmark every configured candidate and write metrics/benchmark.json for DVC."""
    params = load_params()
    dataset, bench = params["dataset"], params["benchmark"]
    rank_by = bench["rank_by"]

    train = _load_training_frame(params)
    groups = train[dataset["group"]] if dataset.get("group") else None
    x, y = split_xy(train, dataset)
    folds = make_folds(y, groups, bench["cv_folds"], params["train"]["random_state"])

    mlflow.set_tracking_uri(get_tracking_uri(params))
    mlflow.set_experiment(params["mlflow"]["experiment_name"])
    EMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    with mlflow.start_run(run_name="benchmark"):
        mlflow.set_tags({"stage": "benchmark", "dataset": dataset["name"]})
        mlflow.log_params(
            {
                "cv_folds": bench["cv_folds"],
                "grouped_folds": groups is not None,
                "threshold": params["evaluate"]["threshold"],
                "n_rows": len(x),
                "n_features": x.shape[1],
                "positive_rate": round(float(y.mean()), 6),
            }
        )

        for name in bench["models"]:
            with mlflow.start_run(run_name=name, nested=True):
                row = _run_candidate(name, params, x, y, folds)
            if row is not None:
                rows.append(row)

        if not rows:
            raise RuntimeError("No candidate completed. Check the skipped runs in MLflow.")

        matrix = add_ensemble_deltas(pd.DataFrame(rows), rank_by)
        matrix = matrix.sort_values(f"{rank_by}_mean", ascending=False).reset_index(drop=True)
        mlflow.log_table(matrix, artifact_file="benchmark_matrix.json")
        mlflow.set_tag("best_candidate", matrix.iloc[0]["candidate"])
        for row in matrix.itertuples(index=False):
            if row.is_ensemble:
                mlflow.log_metric(f"{row.candidate}_gain_{rank_by}", getattr(row, f"gain_{rank_by}_vs_best_single"))
                mlflow.log_metric(f"{row.candidate}_latency_ratio", row.latency_ratio_vs_best_single)

    report = {
        row["candidate"]: {key: value for key, value in row.items() if key not in ("candidate", "is_ensemble")}
        for row in matrix.to_dict(orient="records")
    }
    with open(BENCHMARK_PATH, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    log.info("Ranked by %s:\n%s", rank_by, matrix[["candidate", f"{rank_by}_mean", "latency_ms_per_1k_rows"]])


if __name__ == "__main__":
    main()
