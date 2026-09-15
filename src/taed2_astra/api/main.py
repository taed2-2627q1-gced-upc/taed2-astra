"""FastAPI application exposing the model.

Run locally:  uvicorn taed2_astra.api.main:app --reload
Docs:         http://127.0.0.1:8000/docs
"""

from fastapi import FastAPI, HTTPException

from taed2_astra import __version__
from taed2_astra.api.schemas import HealthResponse, PredictionRequest, PredictionResponse
from taed2_astra.modeling.predict import load_model, predict

app = FastAPI(
    title="Astra Prediction API",
    description="Patient risk prediction service.",
    version=__version__,
)


@app.get("/health", response_model=HealthResponse, tags=["monitoring"])
def health() -> HealthResponse:
    """Report whether the service is up and the model is loadable."""
    try:
        load_model()
        model_loaded = True
    except FileNotFoundError:
        # The service stays up without a model so orchestrators get a clear
        # readiness signal instead of a crash loop.
        model_loaded = False
    return HealthResponse(status="ok", version=__version__, model_loaded=model_loaded)


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict_endpoint(request: PredictionRequest) -> PredictionResponse:
    """Score a batch of records."""
    try:
        results = predict(request.records)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return PredictionResponse(predictions=results)
