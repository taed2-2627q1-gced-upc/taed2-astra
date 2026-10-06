"""Fixtures shared by the test suite."""

import os

import pytest
from fastapi.testclient import TestClient

from taed2_astra.api.main import app

# Tests only write figures to files; a GUI backend (Tk on Windows) warns when torn down off the main thread.
os.environ.setdefault("MPLBACKEND", "Agg")


@pytest.fixture(name="client")
def client_fixture() -> TestClient:
    """A test client for the API."""
    return TestClient(app)
