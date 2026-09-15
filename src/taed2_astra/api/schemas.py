"""Request and response bodies for the API.

Kept separate from the routes so the contract can be reviewed on its own.

The payload is intentionally schema-agnostic: the feature columns are not
fixed until the dataset is chosen. Tighten this into explicit typed fields
once the final column list is known.
"""

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """A batch of records to score."""

    records: list[dict] = Field(..., min_length=1, description="One mapping per record.")


class Prediction(BaseModel):
    """Model output for a single record."""

    risk_probability: float = Field(..., ge=0.0, le=1.0)
    prediction: int


class PredictionResponse(BaseModel):
    """Model output for the whole batch."""

    predictions: list[Prediction]


class HealthResponse(BaseModel):
    """Liveness and readiness of the service."""

    status: str
    version: str
    model_loaded: bool
