"""Stage 4: score the saved model on the held-out test split."""

import mlflow
import numpy as np
import pandas as pd
from codecarbon import EmissionsTracker
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from taed2_astra.config import (
    EMISSIONS_DIR,
    METRICS_DIR,
    METRICS_PATH,
    TEST_PATH,
    get_logger,
    get_tracking_uri,
    load_params,
    write_json,
)
from taed2_astra.energy import emissions_summary
from taed2_astra.features.build_features import split_xy
from taed2_astra.modeling.predict import load_model

log = get_logger(__name__)


def compute_metrics(y_true, y_pred, y_proba) -> dict[str, float]:
    """Return the metrics tracked for every run, as a flat name -> float dict.

    Threshold-free scores (ROC-AUC, PR-AUC, Brier) rank models; the thresholded
    ones describe the operating point the API will actually expose.
    """
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "brier": float(brier_score_loss(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "predicted_positive_rate": float(np.mean(y_pred)),
    }


def main() -> None:
    """Evaluate the model and write metrics.json for DVC to track."""
    params = load_params()
    threshold = params["evaluate"]["threshold"]
    name = params["train"]["model"]

    model = load_model()
    x_test, y_test = split_xy(pd.read_parquet(TEST_PATH), params["dataset"])
    # Inference is the other half of an ML system's energy footprint, next to training.
    EMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    tracker = EmissionsTracker(project_name=f"inference-{name}", output_dir=str(EMISSIONS_DIR), log_level="error")
    tracker.start()
    try:
        proba = model.predict_proba(x_test)[:, 1]
    finally:
        tracker.stop()
    energy, context = emissions_summary(tracker)

    metrics = compute_metrics(y_test, (proba >= threshold).astype(int), proba)
    metrics.update({f"inference_{key}": value for key, value in energy.items()})

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    write_json(METRICS_PATH, metrics)

    mlflow.set_tracking_uri(get_tracking_uri(params))
    mlflow.set_experiment(params["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name=f"evaluate-{name}"):
        mlflow.set_tags({"stage": "evaluate", "candidate": name, "dataset": params["dataset"]["name"]})
        mlflow.log_param("threshold", threshold)
        mlflow.log_param("n_test_rows", len(x_test))
        mlflow.log_metrics(metrics)
        mlflow.set_tags(context)

    log.info("Metrics: %s", metrics)


if __name__ == "__main__":
    main()
