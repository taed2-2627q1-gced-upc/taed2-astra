"""FastAPI application exposing the sepsis early-warning model.

Run locally:  uvicorn taed2_astra.api.main:app --reload
Docs:         http://127.0.0.1:8000/docs
"""

import hashlib
import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Body, FastAPI, Request, Response

from taed2_astra import __version__
from taed2_astra.api.schemas import (
    OPENAPI_EXAMPLES,
    HealthResponse,
    ModelInfo,
    Prediction,
    PredictionRequest,
    PredictionResponse,
    outside_training_range,
)
from taed2_astra.config import METRICS_PATH, MODEL_PATH, get_logger, load_params
from taed2_astra.modeling.predict import load_model, predict

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(api: FastAPI) -> AsyncIterator[None]:
    """Load the model once, before the first request is accepted.

    A missing model or one fitted on a different feature set than the data
    contract stops the server at startup, instead of failing on every request.
    """
    params = load_params()
    model = load_model(MODEL_PATH)
    expected = list(params["validation"]["ranges"])
    if set(model.feature_names_in_) != set(expected):
        raise RuntimeError(
            f"Model features {sorted(model.feature_names_in_)} do not match params.yaml validation.ranges {expected}."
        )
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else {}
    api.state.model = model
    api.state.training_ranges = params["validation"]["ranges"]
    api.state.info = ModelInfo(
        name=params["train"]["model"],
        model_md5=hashlib.md5(MODEL_PATH.read_bytes()).hexdigest(),
        threshold=params["evaluate"]["threshold"],
        features=list(model.feature_names_in_),
        metrics={key: value for key, value in metrics.items() if not key.startswith("inference_")},
    )
    logger.info("Serving %s (md5 %s)", api.state.info.name, api.state.info.model_md5)
    yield


DESCRIPTION = """
Estimates, **every hour and for every ICU patient**, the probability that sepsis is
developing, from the vitals and labs charted in that hour. The model was trained on
the PhysioNet 2019 Computing in Cardiology Challenge data.

### Try it

1. Open **POST /predict** below and click **Try it out**.
2. Pick an example from the **Examples** dropdown (high risk, low risk, extreme value, batch).
3. Click **Execute** and read `risk_probability` and `prediction` in the response.

### Good to know

- Send **one record per patient-hour**. Leave out anything that was not measured; the model fills it in.
- Results come back **in the same order** as the records you sent.
- A record that cannot be a real measurement (a misspelled field, a temperature in °F)
  is rejected with **422**, naming the field. Nothing in that batch is scored.
- **GET /model** tells you exactly which model answered and how well it scored on held-out patients.

*Decision support for research and teaching. Not a certified medical device.*
"""

TAGS = [
    {"name": "inference", "description": "Score patient-hours for sepsis risk."},
    {"name": "monitoring", "description": "Check that the service is up and see which model it serves."},
]

app = FastAPI(
    title="Astra Sepsis Early-Warning API",
    summary="Hourly sepsis risk for ICU patients.",
    description=DESCRIPTION,
    version=__version__,
    openapi_tags=TAGS,
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    """Log method, path, status and latency of every request."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info("%s %s -> %d (%.1f ms)", request.method, request.url.path, response.status_code, elapsed_ms)
    return response


# These docstrings are rendered in /docs, so they are written for API callers.


@app.get(
    "/",
    tags=["monitoring"],
    summary="Welcome message",
    response_description="A welcome message and where the documentation lives.",
)
def root() -> dict:
    """
    Check that you reached the Astra API and find its documentation.

    The interactive documentation you are reading is at `/docs`. The same reference,
    in a print-friendly layout, is at `/redoc`.

    Returns
    -------
    object
        `message`: a welcome text. `docs`: the path of the interactive documentation.
    """
    return {"message": "Astra sepsis early-warning API. See /docs for usage.", "docs": "/docs"}


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["monitoring"],
    summary="Is the service up?",
    response_description="The service is up and ready to score.",
)
def health() -> HealthResponse:
    """
    Check that the service is running and ready to score.

    The service only starts once a valid model is loaded, so a `200` here also means
    `/predict` can answer. The call never touches the model, so it is cheap enough to
    poll every few seconds from a monitor.

    Returns
    -------
    HealthResponse
        `status`: always `"ok"` while the service is up.
        `version`: the version of the API.
    """
    return HealthResponse(status="ok", version=__version__)


@app.get(
    "/model",
    response_model=ModelInfo,
    tags=["monitoring"],
    summary="Which model is answering?",
    response_description="Identity, decision threshold, accepted fields and test scores of the served model.",
)
def model_info(request: Request) -> ModelInfo:
    """
    See which model scores your requests and how well it performed.

    Use `model_md5` to check which model produced a prediction: every `/predict`
    response carries the same value. The scores in `metrics` were measured on ICU
    patients the model never saw during training.

    Returns
    -------
    ModelInfo
        `name`: the type of model, e.g. `hist_gradient_boosting`.
        `model_md5`: fingerprint of the served model.
        `threshold`: a `risk_probability` at or above this gives `prediction = 1`.
        `features`: every field `/predict` accepts.
        `metrics`: test scores such as ROC-AUC, PR-AUC, recall and precision.
    """
    return request.app.state.info


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["inference"],
    summary="Score patient-hours for sepsis risk",
    response_description="One prediction per record, in the same order as the request.",
    responses={
        422: {
            "description": (
                "The request breaks the input rules: an unknown or misspelled field, a value "
                "that cannot be a real measurement, text instead of a number, a missing required "
                "field, or an empty or oversized batch. `detail[].loc` points at the record and "
                "field to fix. Nothing in the batch is scored."
            )
        }
    },
)
def predict_endpoint(
    body: Annotated[PredictionRequest, Body(openapi_examples=OPENAPI_EXAMPLES)], request: Request
) -> PredictionResponse:
    """
    Estimate, for each patient-hour, the probability that sepsis is developing.

    Send one record per patient and hour, with the vitals and labs charted in that
    hour. Each record is scored on its own; the API keeps no patient history. Pick a
    ready-made request from the **Examples** dropdown to try it.

    Parameters
    ----------
    records : list of PatientHour
        The patient-hours to score, at least one and at most `maxItems` per request.

        - Fields charted every hour are required; the `PatientHour` schema marks them.
        - Every other measurement is optional. Leave it out if it was not taken; the
          model fills it in.
        - Values must be possible in the expected unit: temperature in °C, FiO2 as a
          fraction (not a percentage), binary fields as `0` or `1`. Each field's bounds
          are listed in the `PatientHour` schema.

    Returns
    -------
    PredictionResponse
        `predictions`: one entry per record, in the order sent, each with
        `risk_probability` (0 to 1), `prediction` (`1` = alert, when the risk reaches
        `threshold`) and `warnings`.
        `threshold`: the alert threshold applied.
        `model_md5`: fingerprint of the model that answered (see `/model`).

        A warning means a value is possible but rarer than anything the model was
        trained on, for example an HR of 19. The risk is still returned; treat it with care.

    Raises
    ------
    422 Unprocessable Entity
        A record has an unknown or misspelled field, a value that cannot be a real
        measurement, text instead of a number or a missing required field, or the
        batch is empty or too large. `detail[].loc` names the record and field to fix,
        and nothing in the batch is scored.
    """
    # exclude_none: an omitted measurement must reach the imputer as NaN, not as a None object column.
    records = [record.model_dump(exclude_none=True) for record in body.records]
    info: ModelInfo = request.app.state.info
    scores = predict(records, request.app.state.model)
    return PredictionResponse(
        predictions=[
            Prediction(**score, warnings=outside_training_range(record, request.app.state.training_ranges))
            for score, record in zip(scores, records, strict=True)
        ],
        threshold=info.threshold,
        model_md5=info.model_md5,
    )
