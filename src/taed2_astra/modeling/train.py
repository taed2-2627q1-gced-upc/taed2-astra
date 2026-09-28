"""Stage 3: fit the selected candidate, tracking params, metrics and CO2 with MLflow.

Which candidate ships is `train.model` in params.yaml; the benchmark stage is
what informs that choice.
"""

import json
import pickle
from pathlib import Path

import mlflow
import pandas as pd
from codecarbon import EmissionsTracker

from taed2_astra.config import (
    BENCHMARK_PATH,
    EMISSIONS_DIR,
    MODEL_PATH,
    MODELS_DIR,
    TRAIN_PATH,
    get_logger,
    get_tracking_uri,
    load_params,
)
from taed2_astra.energy import emissions_summary
from taed2_astra.features.build_features import split_xy
from taed2_astra.modeling.registry import build_model

log = get_logger(__name__)


def benchmark_best(rank_by: str, path: Path = BENCHMARK_PATH) -> str | None:
    """Return the candidate the benchmark ranked first, or None if it has not run or scored nothing."""
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as handle:
        matrix = json.load(handle)
    key = f"{rank_by}_mean"
    ranked = {name: row[key] for name, row in matrix.items() if key in row}
    return max(ranked, key=ranked.get) if ranked else None


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

        # The benchmark ranks candidates; params.yaml decides which one ships. Record both, or
        # a reader comparing the benchmark's best_candidate tag to models/model.pkl sees a
        # contradiction with no explanation attached.
        best = benchmark_best(params["benchmark"]["rank_by"])
        if best:
            mlflow.set_tags(
                {
                    "benchmark_best": best,
                    "overrides_benchmark": str(best != name).lower(),
                    "selection_rationale": "docs/model_card.md#model-selection",
                }
            )

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
            tracker.stop()

        energy, context = emissions_summary(tracker)
        mlflow.log_metrics(energy)
        mlflow.set_tags(context)
        # MLflow 3 defaults to skops, which rejects make_column_selector; cloudpickle matches models/model.pkl.
        mlflow.sklearn.log_model(
            model, name="model", serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
        )

    with open(MODEL_PATH, "wb") as handle:
        pickle.dump(model, handle)
    log.info(
        "Saved %s to %s (%.6f kg CO2eq, %.6f kWh)", name, MODEL_PATH, energy["emissions_kg_co2"], energy["energy_kwh"]
    )


if __name__ == "__main__":
    main()
