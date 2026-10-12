"""API contract tests. They run on a synthetic model, without data or a trained model on disk."""

import hashlib
import json
import pickle
from http import HTTPStatus

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from taed2_astra.api import main as api_main
from taed2_astra.api.schemas import DEFAULT_EXAMPLE, MAX_BATCH_SIZE, MAX_ERRORS, OPENAPI_EXAMPLES
from taed2_astra.config import load_params
from taed2_astra.modeling.registry import build_model

PARAMS = load_params()
VALIDATION = PARAMS["validation"]
EXAMPLE = DEFAULT_EXAMPLE
LIMITS = PARAMS["api"]["limits"]
# A continuous measurement in the example: the field the malformed-input tests corrupt.
MEASUREMENT = next(name for name in EXAMPLE if name not in VALIDATION["binary"] + VALIDATION["complete"])
# A measurement whose hard limit reaches below its training range: possible, but unseen in training.
EXTREME = next(name for name in EXAMPLE if LIMITS[name][0] < VALIDATION["ranges"][name][0])


def post(client: TestClient, *records: dict):
    """POST a batch of records to /predict."""
    return client.post("/predict", json={"records": list(records)})


def test_root_points_to_the_docs(client):
    """A newcomer hitting the bare URL must learn where the documentation is."""
    response = client.get("/")
    assert response.status_code == HTTPStatus.OK
    assert response.json()["docs"] == "/docs"


def test_health_answers_head_requests(client):
    """Uptime monitors probe with HEAD by default; a 405 there would page someone for a healthy service."""
    assert client.head("/health").status_code == HTTPStatus.OK


def test_health_reports_the_running_version(client):
    """Health is what the VM's service manager and the demo check first."""
    response = client.get("/health")
    assert response.status_code == HTTPStatus.OK
    assert response.json()["status"] == "ok"


def test_model_endpoint_identifies_the_served_file(client, served_model_path):
    """The MD5 must match the file on disk, so a prediction can be traced back to a dvc.lock entry."""
    info = client.get("/model").json()
    assert info["model_md5"] == hashlib.md5(served_model_path.read_bytes()).hexdigest()
    assert info["threshold"] == PARAMS["evaluate"]["threshold"]
    assert set(info["features"]) == set(VALIDATION["ranges"])


def test_documented_example_is_scored(client):
    """The example shown in /docs must be accepted, or the first thing a new user tries fails."""
    response = post(client, EXAMPLE)
    assert response.status_code == HTTPStatus.OK
    body = response.json()
    (prediction,) = body["predictions"]
    assert 0.0 <= prediction["risk_probability"] <= 1.0
    assert prediction["prediction"] == int(prediction["risk_probability"] >= body["threshold"])
    assert body["model_md5"] == client.get("/model").json()["model_md5"]


@pytest.mark.parametrize("name", list(OPENAPI_EXAMPLES))
def test_every_dropdown_example_in_docs_is_scored(client, name):
    """Each example offered under "Try it out" must succeed, or the docs teach a request that fails."""
    response = client.post("/predict", json=OPENAPI_EXAMPLES[name]["value"])
    assert response.status_code == HTTPStatus.OK
    assert len(response.json()["predictions"]) == len(OPENAPI_EXAMPLES[name]["value"]["records"])


@pytest.mark.parametrize("internal", ["params.yaml", "dvc.lock", ".pkl", ".py", "app.state"])
def test_docs_speak_to_callers_not_to_the_code(client, internal):
    """/docs is read by people calling the API, who never see the repository behind it."""
    assert internal not in client.get("/openapi.json").text


def test_only_the_always_charted_fields_are_required(client):
    """At the bedside most labs are missing; a record with just the complete fields must score."""
    minimal = {name: EXAMPLE[name] for name in VALIDATION["complete"]}
    assert post(client, minimal).status_code == HTTPStatus.OK


def test_a_field_present_in_some_records_only_is_imputed(client):
    """A measurement taken for one patient but not another must not break the batch."""
    without = {key: value for key, value in EXAMPLE.items() if key != MEASUREMENT}
    response = post(client, EXAMPLE, without)
    assert response.status_code == HTTPStatus.OK
    assert len(response.json()["predictions"]) == 2


def test_predictions_follow_request_order(client):
    """Clients match results to patients by position, so order must be preserved."""
    driver, (low, high) = next(iter(VALIDATION["ranges"].items()))
    calm, critical = {**EXAMPLE, driver: low}, {**EXAMPLE, driver: high}
    forward = [p["risk_probability"] for p in post(client, calm, critical).json()["predictions"]]
    backward = [p["risk_probability"] for p in post(client, critical, calm).json()["predictions"]]
    assert forward[1] > forward[0]
    assert forward == backward[::-1]


@pytest.mark.parametrize(
    ("case", "record"),
    [
        ("misspelled field", {**EXAMPLE, MEASUREMENT.lower(): EXAMPLE[MEASUREMENT]}),
        ("value above its hard limit", {**EXAMPLE, MEASUREMENT: LIMITS[MEASUREMENT][1] + 1}),
        ("value below its hard limit", {**EXAMPLE, MEASUREMENT: LIMITS[MEASUREMENT][0] - 1}),
        ("number sent as text", {**EXAMPLE, MEASUREMENT: str(EXAMPLE[MEASUREMENT])}),
        ("binary field outside 0/1", {**EXAMPLE, VALIDATION["binary"][0]: 2}),
        ("always-charted field missing", {k: v for k, v in EXAMPLE.items() if k != VALIDATION["complete"][0]}),
    ],
)
def test_malformed_records_are_rejected_not_scored(client, case, record):
    """Each of these would otherwise be imputed or coerced and scored silently as a different patient."""
    response = post(client, record)
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY, case


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_non_finite_numbers_are_rejected_with_a_422(client, literal):
    """Python's JSON parser accepts NaN and Infinity; echoing them back in the error used to crash with a 500."""
    # Written by hand: json.dumps cannot produce 1e999, and the test must not depend on its NaN spelling.
    body = json.dumps({"records": [{**EXAMPLE, MEASUREMENT: "PLACEHOLDER"}]}).replace('"PLACEHOLDER"', literal)
    response = client.post("/predict", content=body, headers={"Content-Type": "application/json"})
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert MEASUREMENT in response.json()["detail"][0]["loc"]


def test_unknown_top_level_keys_are_rejected(client):
    """A misspelled key next to records would otherwise be dropped without the client noticing."""
    response = client.post("/predict", json={"records": [EXAMPLE], "threshold": 0.1})
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


def test_errors_name_the_offending_field(client):
    """A clinician integrating the API must see which field to fix, not just that something failed."""
    response = post(client, {**EXAMPLE, MEASUREMENT: LIMITS[MEASUREMENT][1] + 1})
    assert MEASUREMENT in response.json()["detail"][0]["loc"]


def test_extreme_but_possible_values_are_scored_with_a_warning(client):
    """The sickest patients fall outside the training range; refusing them would fail exactly when it matters."""
    response = post(client, {**EXAMPLE, EXTREME: LIMITS[EXTREME][0]})
    assert response.status_code == HTTPStatus.OK
    (prediction,) = response.json()["predictions"]
    assert len(prediction["warnings"]) == 1
    assert prediction["warnings"][0].startswith(f"{EXTREME}=")


def test_in_range_records_carry_no_warnings(client):
    """Warnings must stay rare and meaningful, or clients learn to ignore them."""
    (prediction,) = post(client, EXAMPLE).json()["predictions"]
    assert prediction["warnings"] == []


def swapped(low: str, high: str) -> dict:
    """Return the example with low set above high, both inside their hard limits."""
    top = min(LIMITS[low][1], LIMITS[high][1])
    bottom = max(LIMITS[low][0], LIMITS[high][0])
    return {**EXAMPLE, low: top, high: bottom}


@pytest.mark.parametrize(("low", "high"), PARAMS["api"]["ordered"])
def test_contradicting_fields_are_scored_with_a_warning(client, low, high):
    """Each value alone is plausible, but no patient has, say, a diastolic above the systolic pressure."""
    response = post(client, swapped(low, high))
    assert response.status_code == HTTPStatus.OK
    (prediction,) = response.json()["predictions"]
    assert any(low in warning and high in warning for warning in prediction["warnings"])


@pytest.mark.parametrize("value", [0, 1])
@pytest.mark.parametrize("group", PARAMS["api"]["one_hot"])
def test_one_hot_groups_need_exactly_one_flag(client, group, value):
    """A patient sits in exactly one ICU type; all flags set, or none, is a broken export."""
    (prediction,) = post(client, {**EXAMPLE, **dict.fromkeys(group, value)}).json()["predictions"]
    assert any(all(name in warning for name in group) for warning in prediction["warnings"])


def test_relation_rules_name_real_features():
    """A typo in a rule would silently switch that check off."""
    rules = PARAMS["api"]["ordered"] + PARAMS["api"]["one_hot"]
    assert {name for rule in rules for name in rule} <= set(VALIDATION["ranges"])


def test_validation_errors_are_capped(client):
    """Thousands of unknown keys must not turn a small request into a huge answer."""
    record = {**EXAMPLE, **{f"unknown_{i}": 1 for i in range(MAX_ERRORS * 5)}}
    assert len(post(client, record).json()["detail"]) == MAX_ERRORS


def test_long_rejected_input_is_not_echoed_back(client):
    """A 1 MB field name is rejected; repeating it in the error would double the cost of the request."""
    response = post(client, {**EXAMPLE, "k" * 1_000_000: 1})
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert len(response.content) < 10_000


def test_hard_limits_cover_every_training_value():
    """Serving must accept everything the model was trained on, for every feature it was trained on."""
    assert set(LIMITS) == set(VALIDATION["ranges"])
    for name, (low, high) in VALIDATION["ranges"].items():
        assert LIMITS[name][0] <= low and high <= LIMITS[name][1], name


@pytest.mark.parametrize("size", [0, MAX_BATCH_SIZE + 1])
def test_batch_size_is_bounded(client, size):
    """An empty batch is a client bug; an oversized one could exhaust the VM's memory."""
    assert post(client, *[EXAMPLE] * size).status_code == HTTPStatus.UNPROCESSABLE_ENTITY


def test_server_does_not_start_without_a_model(tmp_path, monkeypatch):
    """A server that cannot predict must fail at startup, not answer every request with an error."""
    monkeypatch.setattr(api_main, "MODEL_PATH", tmp_path / "missing.pkl")
    with pytest.raises(FileNotFoundError), TestClient(api_main.app):
        pass


def test_server_does_not_start_with_a_model_for_other_features(tmp_path, monkeypatch):
    """A model fitted on other columns would score every request on imputed values only."""
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(size=(200, 3)), columns=["a", "b", "c"])
    model = build_model(PARAMS["train"]["model"], PARAMS["train"]).fit(x, (x["a"] > 0).astype(int))
    path = tmp_path / "other.pkl"
    path.write_bytes(pickle.dumps(model))
    monkeypatch.setattr(api_main, "MODEL_PATH", path)
    with pytest.raises(RuntimeError, match="do not match"), TestClient(api_main.app):
        pass
