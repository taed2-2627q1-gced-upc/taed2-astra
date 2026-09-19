"""Stage 3: fit the selected candidate, tracking params, metrics and CO2 with MLflow.

Which candidate ships is `train.model` in params.yaml; the benchmark stage is
what informs that choice.
"""

import pickle

import mlflow
import pandas as pd
from codecarbon import EmissionsTracker

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
from taed2_astra.modeling.registry import build_model

log = get_logger(__name__)


def main() -> None:
    """Fit the selected model on the training split and save it to models/."""
    params = load_params()
    dataset, train_params = params["dataset"], params["train"]
    name = train_params["model"]

    mlflow.set_tracking_uri(get_tracking_uri(params))
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    x_train, y_train = split_xy(pd.read_parquet(TRAIN_PATH), dataset)
    model = build_model(name, train_params)

    EMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=f"train-{name}"):
        mlflow.set_tags({"stage": "train", "candidate": name, "dataset": dataset["name"]})
        mlflow.log_params(
            {"model": name, "random_state": train_params["random_state"], **train_params["candidates"][name]}
        )
        mlflow.log_param("n_features", x_train.shape[1])
        mlflow.log_param("n_train_rows", len(x_train))

        tracker = EmissionsTracker(project_name=f"train-{name}", output_dir=str(EMISSIONS_DIR), log_level="error")
        tracker.start()
        try:
            model.fit(x_train, y_train)
        finally:
            emissions = tracker.stop()

        mlflow.log_metric("emissions_kg_co2", emissions or 0.0)
        mlflow.sklearn.log_model(model, name="model")

    with open(MODEL_PATH, "wb") as handle:
        pickle.dump(model, handle)
    log.info("Saved %s to %s (%.6f kg CO2eq)", name, MODEL_PATH, emissions or 0.0)


if __name__ == "__main__":
    main()
