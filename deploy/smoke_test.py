"""Smoke test against a running API: the same sequence we show in the demo.

    uv run python deploy/smoke_test.py http://127.0.0.1:8000

Uses only the standard library so it runs on the VM without dev dependencies.
Exits non-zero on the first check that fails.
"""

import json
import sys
import urllib.error
import urllib.request
from http import HTTPStatus

from taed2_astra.config import load_params


def call(url: str, payload: dict | None = None) -> tuple[int, dict]:
    """Return the status code and JSON body of a GET, or of a POST when a payload is given."""
    data = None if payload is None else json.dumps(payload).encode()
    # Cloudflare in front of the public host rejects urllib's default User-Agent with a 403 (error 1010).
    headers = {"Content-Type": "application/json", "User-Agent": "astra-smoke-test"}
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


def check(name: str, ok: bool, detail: object) -> None:
    """Print one result line and stop at the first failure."""
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        sys.exit(1)


def main(base_url: str) -> None:
    """Run health, model, a valid prediction and a rejected one against base_url."""
    base_url = base_url.rstrip("/")
    params = load_params()
    example = next(iter(params["api"]["examples"].values()))["record"]

    status, body = call(f"{base_url}/health")
    check("GET /health", status == HTTPStatus.OK, body)

    status, body = call(f"{base_url}/model")
    check("GET /model", status == HTTPStatus.OK, {key: body.get(key) for key in ("name", "model_md5", "threshold")})

    status, body = call(f"{base_url}/predict", {"records": [example]})
    check("POST /predict (documented example)", status == HTTPStatus.OK, body.get("predictions"))

    limits, ranges = params["api"]["limits"], params["validation"]["ranges"]
    field = next(name for name in example if limits[name][0] < ranges[name][0])
    status, body = call(f"{base_url}/predict", {"records": [{**example, field: limits[field][0]}]})
    scored = status == HTTPStatus.OK and bool(body["predictions"][0]["warnings"])
    check(f"POST /predict ({field}={limits[field][0]} is scored with a warning)", scored, body)

    status, body = call(f"{base_url}/predict", {"records": [{**example, field: limits[field][1] + 1}]})
    rejected = status == HTTPStatus.UNPROCESSABLE_ENTITY
    check(f"POST /predict ({field} beyond its hard limit is rejected)", rejected, status)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000")
