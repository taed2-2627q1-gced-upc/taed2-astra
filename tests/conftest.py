"""Fixtures shared by the test suite."""

import os
import pickle
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from taed2_astra.api import main as api_main
from taed2_astra.config import load_params
from taed2_astra.modeling.registry import build_model

# Tests only write figures to files; a GUI backend (Tk on Windows) warns when torn down off the main thread.
os.environ.setdefault("MPLBACKEND", "Agg")


def synthetic_patient_hours(validation: dict, rows: int, seed: int = 0) -> pd.DataFrame:
    """Return random patient-hours inside the contract's ranges, with labs missing like at the bedside."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            name: rng.integers(low, high + 1, rows) if name in validation["binary"] else rng.uniform(low, high, rows)
            for name, (low, high) in validation["ranges"].items()
        }
    )
    for name in frame.columns.difference(validation["complete"]):
        frame.loc[rng.random(rows) < 0.3, name] = np.nan
    return frame


@pytest.fixture(name="served_model_path", scope="session")
def served_model_path_fixture(tmp_path_factory) -> Path:
    """The shipped candidate fitted on synthetic data with the real feature set, saved to disk.

    The first feature drives the label, so tests can check that risk follows an input.
    """
    params = load_params()
    x = synthetic_patient_hours(params["validation"], rows=600)
    driver = x.columns[0]
    y = (x[driver] > x[driver].quantile(0.7)).astype(int)
    model = build_model(params["train"]["model"], params["train"]).fit(x, y)
    path = tmp_path_factory.mktemp("model") / "model.pkl"
    path.write_bytes(pickle.dumps(model))
    return path


@pytest.fixture(name="client")
def client_fixture(served_model_path, monkeypatch, tmp_path) -> Iterator[TestClient]:
    """A test client whose startup loaded the synthetic model.

    The `with` block runs the lifespan, exactly as uvicorn does before serving.
    """
    monkeypatch.setattr(api_main, "MODEL_PATH", served_model_path)
    monkeypatch.setattr(api_main, "ACCESS_LOG_PATH", tmp_path / "logs" / "access.log")
    with TestClient(api_main.app) as client:
        yield client
