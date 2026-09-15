"""Fixtures shared by the test suite."""

import pytest
from fastapi.testclient import TestClient

from taed2_astra.api.main import app


@pytest.fixture(name="client")
def client_fixture() -> TestClient:
    """A test client for the API."""
    return TestClient(app)
