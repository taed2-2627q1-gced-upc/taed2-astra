"""Green-ML tracking and reporting with CodeCarbon, as practised in the course lab.

Tracking: each compute-heavy stage wraps its work in CodeCarbon's
``EmissionsTracker`` (``tracker.start()`` ... ``tracker.stop()``), which appends one
row to reports/emissions/emissions.csv with the duration (s), the emissions (kg
CO2eq) and the energy consumed (kWh), plus where and on which hardware it ran.
``emissions_summary`` turns that row into MLflow metrics and tags.

Reporting: ``main`` writes the shipped model's training footprint into the
Hugging Face ``co2_eq_emissions`` metadata at the top of docs/model_card.md.
"""

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import yaml
from codecarbon import EmissionsTracker

from taed2_astra.config import (
    EMISSIONS_PATH,
    METRICS_PATH,
    MODEL_CARD_PATH,
    MODEL_PATH,
    TRAIN_PATH,
    get_logger,
    load_params,
)

log = get_logger(__name__)

REPORTED_METRICS = ("roc_auc", "pr_auc", "recall", "precision", "f1", "brier")


def emissions_summary(tracker: EmissionsTracker) -> tuple[dict[str, float], dict[str, str]]:
    """Return a stopped tracker's emissions.csv quantities as metrics, and its run context as tags.

    Metrics are the three quantities CodeCarbon logs per experiment; tags record the
    location and hardware the model card must report next to them.
    """
    data = tracker.final_emissions_data
    metrics = {
        "duration_s": float(data.duration),
        "emissions_kg_co2": float(data.emissions),
        "energy_kwh": float(data.energy_consumed),
    }
    tags = {
        "cc_country": str(data.country_name),
        "cc_region": str(data.region),
        "cc_cpu_model": str(data.cpu_model),
        "cc_cpu_count": str(data.cpu_count),
        "cc_gpu_model": str(data.gpu_model),
        "cc_codecarbon_version": str(data.codecarbon_version),
    }
    return metrics, tags


def latest_run(project_name: str, csv_path: Path = EMISSIONS_PATH) -> pd.Series:
    """Return the most recent emissions.csv row written under ``project_name``."""
    runs = pd.read_csv(csv_path)
    runs = runs[runs["project_name"] == project_name]
    if runs.empty:
        raise LookupError(f"No '{project_name}' row in {csv_path}. Run `dvc repro train` first.")
    return runs.iloc[-1]


def _hardware(run: pd.Series) -> str:
    """Describe the compute the way the model card asks: how much and what kind."""
    hardware = f"{int(run['cpu_count'])} x {run['cpu_model']} (CPU)"
    gpu_count = run.get("gpu_count")
    if pd.notna(gpu_count) and gpu_count:
        hardware += f", {int(gpu_count)} x {run['gpu_model']} (GPU)"
    return hardware


def build_card_metadata(
    run: pd.Series, metrics: dict[str, float], model_file_size: int, datasets_size: int, card: dict
) -> dict:
    """Return the co2_eq_emissions and model_info blocks of the Hugging Face model card metadata.

    Args:
        run: The shipped model's training row from emissions.csv.
        metrics: Test-split metrics from metrics/metrics.json.
        model_file_size: Size of models/model.pkl in bytes.
        datasets_size: Number of training rows.
        card: The ``model_card`` section of params.yaml (facts CodeCarbon cannot measure).
    """
    location = ", ".join(str(part) for part in (run.get("region"), run.get("country_name")) if pd.notna(part))
    return {
        "co2_eq_emissions": {
            "emissions": round(float(run["emissions"]) * 1000.0, 4),  # the card uses grams
            "power_consumption": round(float(run["energy_consumed"]), 6),  # kWh
            "source": f"CodeCarbon {run['codecarbon_version']} (EmissionsTracker), reports/emissions/emissions.csv",
            "training_type": card["training_type"],
            "geographical_location": location,
            "hardware_used": _hardware(run),
            "training_time": round(float(run["duration"]), 1),  # seconds
            "optimization_techniques": card["optimization_techniques"],
        },
        "model_info": {
            "model_file_size": model_file_size,
            "datasets_size": datasets_size,
            "performance_metrics": [
                {"metric": name, "value": round(metrics[name], 4)} for name in REPORTED_METRICS if name in metrics
            ],
        },
    }


def write_front_matter(path: Path, metadata: dict) -> None:
    """Merge ``metadata`` into the YAML front matter of a markdown file, keeping its body and other keys."""
    text = path.read_text(encoding="utf-8")
    front: dict = {}
    body = text
    if text.startswith("---\n"):
        end = text.index("\n---\n", 4)
        front = yaml.safe_load(text[4:end]) or {}
        body = text[end + len("\n---\n") :].lstrip("\n")
    front.update(metadata)
    dumped = yaml.safe_dump(front, sort_keys=False, allow_unicode=True, width=100)
    path.write_text(f"---\n{dumped}---\n\n{body}", encoding="utf-8", newline="\n")  # LF, as Git stores it


def main() -> None:
    """Write the shipped model's training footprint into the model card metadata."""
    params = load_params()
    run = latest_run(f"train-{params['train']['model']}")
    with open(METRICS_PATH, encoding="utf-8") as handle:
        metrics = json.load(handle)
    metadata = build_card_metadata(
        run,
        metrics,
        model_file_size=MODEL_PATH.stat().st_size,
        datasets_size=pq.ParquetFile(TRAIN_PATH).metadata.num_rows,
        card=params["model_card"],
    )
    write_front_matter(MODEL_CARD_PATH, metadata)
    log.info("Model card co2_eq_emissions updated: %s", metadata["co2_eq_emissions"])


if __name__ == "__main__":
    main()
