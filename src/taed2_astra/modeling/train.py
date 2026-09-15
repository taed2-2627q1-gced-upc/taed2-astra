"""Stage 3: train the model, tracking params, metrics and CO2 with MLflow.

TODO(team): implement build_model() and the fit call.
"""

import pickle

import mlflow
import pandas as pd
from codecarbon import EmissionsTracker
from sklearn.pipeline import Pipeline

from taed2_astra.config import (
    EMISSIONS_DIR,
    MODEL_PATH,
    MODELS_DIR,
    TRAIN_PATH,
    get_logger,
    get_tracking_uri,
    load_params,
)
from taed2_astra.features.build_features import split_xy

log = get_logger(__name__)


def build_model(train_params: dict) -> Pipeline:
    """Return an unfitted preprocessing + estimator pipeline.

    Keeping preprocessing inside the pipeline means the API loads one object
    and cannot forget a transformation step.
    """
    raise NotImplementedError("Build the Pipeline from train_params")


def main() -> None:
    """Fit the model on the training split and save it to models/."""
    params = load_params()

    mlflow.set_tracking_uri(get_tracking_uri(params))
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    x_train, y_train = split_xy(pd.read_parquet(TRAIN_PATH), params["dataset"]["target"])
    model = build_model(params["train"])

    EMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name="train"):
        mlflow.log_params(params["train"])
        mlflow.log_param("n_features", x_train.shape[1])
        mlflow.log_param("n_train_rows", len(x_train))

        tracker = EmissionsTracker(output_dir=str(EMISSIONS_DIR), log_level="error")
        tracker.start()
        try:
            model.fit(x_train, y_train)
        finally:
            emissions = tracker.stop()

        mlflow.log_metric("emissions_kg_co2", emissions or 0.0)
        mlflow.sklearn.log_model(model, name="model")

    with open(MODEL_PATH, "wb") as handle:
        pickle.dump(model, handle)
    log.info("Saved model to %s (%.6f kg CO2eq)", MODEL_PATH, emissions or 0.0)


if __name__ == "__main__":
    main()
