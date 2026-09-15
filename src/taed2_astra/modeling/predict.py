"""Load the trained model and turn records into predictions.

Used by both the evaluation stage and the API so they cannot diverge.

TODO(team): implement predict().
"""

import pickle
from functools import lru_cache
from pathlib import Path

from sklearn.pipeline import Pipeline

from taed2_astra.config import MODEL_PATH


@lru_cache(maxsize=1)
def load_model(path: Path = MODEL_PATH) -> Pipeline:
    """Return the fitted pipeline, loading it from disk only once."""
    if not path.exists():
        raise FileNotFoundError(f"No model at {path}. Run `dvc repro` or `dvc pull` first.")
    with open(path, "rb") as handle:
        return pickle.load(handle)


def predict(records: list[dict]) -> list[dict]:
    """Score input records and return one result dict per record."""
    raise NotImplementedError("Align the records to the model features and predict")
