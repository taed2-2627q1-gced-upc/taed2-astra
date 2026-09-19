"""Load the trained model and turn records into predictions.

Used by both the evaluation stage and the API so they cannot diverge.
"""

import pickle
from functools import lru_cache
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

from taed2_astra.config import MODEL_PATH, load_params


@lru_cache(maxsize=1)
def load_model(path: Path = MODEL_PATH) -> Pipeline:
    """Return the fitted pipeline, loading it from disk only once."""
    if not path.exists():
        raise FileNotFoundError(f"No model at {path}. Run `dvc repro` or `dvc pull` first.")
    with open(path, "rb") as handle:
        return pickle.load(handle)


def predict(records: list[dict]) -> list[dict]:
    """Score input records and return one result dict per record.

    Records are aligned to the columns the model was fitted on: unknown keys are
    dropped and missing ones become NaN, which the pipeline's imputer handles.
    """
    model = load_model()
    threshold = load_params()["evaluate"]["threshold"]
    frame = pd.DataFrame.from_records(records).reindex(columns=list(model.feature_names_in_))
    proba = model.predict_proba(frame)[:, 1]
    return [{"risk_probability": float(p), "prediction": int(p >= threshold)} for p in proba]
