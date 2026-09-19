"""Project paths and parameters.

Single place that knows the repository layout. Import from here instead of
building paths by hand, so moving a folder is a one-line change.
"""

import logging
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Credentials live in .env (gitignored), never in a tracked file.
load_dotenv()

# config.py -> taed2_astra -> src -> repository root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "metrics"
REPORTS_DIR = PROJECT_ROOT / "reports"
EMISSIONS_DIR = REPORTS_DIR / "emissions"
FIGURES_DIR = REPORTS_DIR / "figures"
VALIDATION_PATH = REPORTS_DIR / "data_validation.json"

PARAMS_PATH = PROJECT_ROOT / "params.yaml"
MODEL_PATH = MODELS_DIR / "model.pkl"

# Artefacts written by the pipeline, in the order the stages produce them.
TRAIN_PATH = PROCESSED_DATA_DIR / "train.parquet"
TEST_PATH = PROCESSED_DATA_DIR / "test.parquet"
METRICS_PATH = METRICS_DIR / "metrics.json"
BENCHMARK_PATH = METRICS_DIR / "benchmark.json"


def load_params(path: Path = PARAMS_PATH) -> dict:
    """Return params.yaml as a plain dict."""
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def get_tracking_uri(params: dict) -> str:
    """Return the MLflow tracking URI.

    MLFLOW_TRACKING_URI wins over params.yaml so a teammate can log runs
    locally without editing a tracked file (which would invalidate DVC stages).
    """
    return os.environ.get("MLFLOW_TRACKING_URI") or params["mlflow"]["tracking_uri"]


def get_logger(name: str) -> logging.Logger:
    """Return a logger that prints to stderr with a consistent format."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)
