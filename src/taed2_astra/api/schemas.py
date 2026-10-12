"""Request and response bodies for the API.

Kept separate from the routes so the contract can be reviewed on its own.

The record schema is generated from `params.yaml`, so no column name is hardcoded
here. Field types and required fields come from `validation`, the data contract the
`validate` stage enforces on the training data. Bounds come from `api.limits`, which
are wider: serving must score extreme but real patients that training rarely saw.
"""

from pydantic import BaseModel, ConfigDict, Field, create_model

from taed2_astra.config import load_params


def build_record_model(validation: dict, limits: dict, example: dict | None = None) -> type[BaseModel]:
    """Return a Pydantic model for one patient-hour, typed by the data contract and bounded by `limits`.

    Features in `complete` are always charted, so they are required; every other
    measurement is optional because most labs are absent in any given hour.
    Unknown fields are rejected: a misspelled key would otherwise be dropped and
    silently scored as a missing measurement.
    """
    fields = {}
    for name in validation["ranges"]:
        low, high = limits[name]
        kind = int if name in validation["binary"] else float
        bounds = {"ge": low, "le": high}
        if name in validation["complete"]:
            fields[name] = (kind, Field(..., **bounds))
        else:
            fields[name] = (kind | None, Field(None, **bounds))
    # strict: a string such as "98" is a client bug, not a heart rate to coerce.
    # allow_inf_nan: Python's JSON parser accepts NaN and Infinity, which are not measurements.
    config = ConfigDict(
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
        json_schema_extra={"examples": [example]} if example else None,
    )
    return create_model("PatientHour", __config__=config, **fields)


def outside_validated_range(record: dict, ranges: dict) -> list[str]:
    """Return one warning per measurement outside the range the training data was checked against.

    Inside the range does not mean seen in training: the ranges are wide plausibility
    bounds, so the warning flags values that are rare in training, not every unseen one.
    """
    return [
        f"{name}={value} is outside the validated range [{ranges[name][0]}, {ranges[name][1]}]; "
        "such values are rare in the training data, so the risk is less reliable"
        for name, value in record.items()
        if not ranges[name][0] <= value <= ranges[name][1]
    ]


def inconsistencies(record: dict, ordered: list, one_hot: list) -> list[str]:
    """Return one warning per relation between fields that the record breaks.

    Each field is checked on its own by the schema; this catches combinations that no
    real patient has, such as a diastolic pressure above the systolic. A rule is only
    checked when every field in it was sent.
    """
    warnings = [
        f"{low}={record[low]} is above {high}={record[high]}; this is usually a charting error"
        for low, high in ordered
        if low in record and high in record and record[low] > record[high]
    ]
    warnings += [
        f"exactly one of {', '.join(group)} should be 1; this is usually a charting error"
        for group in one_hot
        if all(name in record for name in group) and sum(record[name] for name in group) != 1
    ]
    return warnings


def build_openapi_examples(examples: dict) -> dict:
    """Return the named request bodies for the /docs dropdown, plus one batch holding every example."""
    named = {
        key: {
            "summary": example["summary"],
            "description": example["description"],
            "value": {"records": [example["record"]]},
        }
        for key, example in examples.items()
    }
    named["batch"] = {
        "summary": "Batch - every example at once",
        "description": "One prediction per record, returned in the same order as the records.",
        "value": {"records": [example["record"] for example in examples.values()]},
    }
    return named


_PARAMS = load_params()
DEFAULT_EXAMPLE: dict = next(iter(_PARAMS["api"]["examples"].values()))["record"]
OPENAPI_EXAMPLES = build_openapi_examples(_PARAMS["api"]["examples"])
PatientHour = build_record_model(_PARAMS["validation"], _PARAMS["api"]["limits"], DEFAULT_EXAMPLE)
MAX_BATCH_SIZE: int = _PARAMS["api"]["max_batch_size"]
MAX_ERRORS: int = _PARAMS["api"]["max_errors"]


class PredictionRequest(BaseModel):
    """A batch of patient-hours to score."""

    # A misspelled "record" key must fail loudly, as a misspelled field inside a record does.
    model_config = ConfigDict(extra="forbid")

    records: list[PatientHour] = Field(  # type: ignore[valid-type]
        ...,
        min_length=1,
        max_length=MAX_BATCH_SIZE,
        description="One object per patient-hour. Omit measurements that were not taken.",
    )


class Prediction(BaseModel):
    """Model output for a single patient-hour."""

    risk_probability: float = Field(..., ge=0.0, le=1.0, description="Estimated probability of sepsis.")
    prediction: int = Field(..., description="1 if risk_probability >= threshold, else 0.")
    warnings: list[str] = Field(
        default_factory=list,
        description=("Values that are rare in training, or fields that contradict each other. Empty for most records."),
    )


class PredictionResponse(BaseModel):
    """Model output for the whole batch, in request order."""

    predictions: list[Prediction]
    threshold: float = Field(..., description="Decision threshold applied to risk_probability.")
    model_md5: str = Field(..., description="Fingerprint of the model that answered; see /model.")


class ModelInfo(BaseModel):
    """What is being served and how well it scored on the held-out test split."""

    name: str
    model_md5: str
    threshold: float
    features: list[str]
    metrics: dict[str, float]


class HealthResponse(BaseModel):
    """Liveness of the service. It only starts once the model is loaded."""

    status: str
    version: str
