"""Stage 4: score the saved model on the held-out test split.

TODO(team): implement compute_metrics().
"""

import json

import mlflow
import pandas as pd

from taed2_astra.config import (
    METRICS_DIR,
    METRICS_PATH,
    TEST_PATH,
    get_logger,
    get_tracking_uri,
    load_params,
)
from taed2_astra.features.build_features import split_xy
from taed2_astra.modeling.predict import load_model

log = get_logger(__name__)


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    """Return the metrics tracked for every run, as a flat name -> float dict."""
    raise NotImplementedError("Compute the evaluation metrics")


def main() -> None:
    """Evaluate the model and write metrics.json for DVC to track."""
    params = load_params()

    model = load_model()
    x_test, y_test = split_xy(pd.read_parquet(TEST_PATH), params["dataset"]["target"])
    metrics = compute_metrics(
        y_test,
        model.predict(x_test),
        model.predict_proba(x_test)[:, 1],
    )

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    mlflow.set_tracking_uri(get_tracking_uri(params))
    mlflow.set_experiment(params["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name="evaluate"):
        mlflow.log_metrics(metrics)

    log.info("Metrics: %s", metrics)


if __name__ == "__main__":
    main()
