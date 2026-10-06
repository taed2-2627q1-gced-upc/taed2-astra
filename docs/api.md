# API reference

The API serves the model trained by the DVC pipeline (`models/model.pkl`). Interactive
documentation is generated at `/docs` (Swagger UI) and `/redoc` on every running instance.
In `/docs`, open **POST /predict**, click **Try it out** and pick a request from the
**Examples** dropdown: high risk, low risk, required fields only, an extreme value that
comes back with a warning, or all of them as one batch. The examples live in
`params.yaml: api.examples`, and a test checks that each one is accepted.

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/` | Welcome message and a pointer to `/docs` |
| `GET` | `/health` | `{"status": "ok", "version": "..."}` while the service is up |
| `GET` | `/model` | Name, MD5, threshold, feature list and test-split metrics of the served model |
| `POST` | `/predict` | Sepsis risk for a batch of patient-hours |

### `POST /predict`

Request: a list of records, one per patient-hour, between 1 and `api.max_batch_size`
(1000) records.

```json
{
  "records": [
    {"Hour": 5, "HR": 104, "O2Sat": 94, "Temp": 38.6, "MAP": 68, "Resp": 26, "FiO2": 0.4,
     "Lactate": 3.1, "Age": 67, "Gender": 1, "Unit1": 1, "Unit2": 0, "HospAdmTime": -12.5, "ICULOS": 6}
  ]
}
```

Response (illustrative values): one prediction per record, **in request order**, plus what produced them.

```json
{
  "predictions": [{"risk_probability": 0.31, "prediction": 0, "warnings": []}],
  "threshold": 0.5,
  "model_md5": "4728f865299ee181f789f2f36b0a88ea"
}
```

| Field | Meaning |
|-------|---------|
| `risk_probability` | Estimated probability that the patient is in the sepsis window, in [0, 1] |
| `prediction` | `1` if `risk_probability >= threshold`, else `0` |
| `warnings` | Measurements outside the range seen in training (see below). Empty for most records |
| `threshold` | `params.yaml: evaluate.threshold`, the same one used for the reported metrics |
| `model_md5` | MD5 of the served file. It matches `models/model.pkl` in `dvc.lock`, so every prediction can be traced back to a commit |

## Examples

Ready-to-send patient-hours for trying the API or running a demo. The scores are what
the model with MD5 `4728f865…` returns. A retrained model gives slightly different numbers.

| Patient-hour | Clinical picture | Risk | `prediction` |
|--------------|------------------|------|--------------|
| Stable | Normal vitals and labs, room air, 3 h into the ICU stay | 0.05 | `0` |
| Stable, minimal | Only the required fields plus a normal HR and temperature | 0.02 | `0` |
| Extreme bradycardia | HR 19 with fever: scored, with a warning | 0.19 | `0` |
| Deteriorating | Fever, tachycardia, fast breathing, low MAP, lactate 3.1, on oxygen (the `/docs` example) | 0.86 | `1` |

### Low risk: `prediction = 0`

```json
{
  "records": [
    {"Hour": 2, "HR": 72, "O2Sat": 98, "Temp": 36.8, "SBP": 125, "MAP": 88, "DBP": 70, "Resp": 15,
     "FiO2": 0.21, "Lactate": 1.0, "WBC": 7.5, "Creatinine": 0.9, "Platelets": 250,
     "Age": 45, "Gender": 0, "Unit1": 0, "Unit2": 1, "HospAdmTime": -2.0, "ICULOS": 3}
  ]
}
```

The minimal version, with only the required fields plus HR and temperature:

```json
{"records": [{"Hour": 2, "HR": 72, "Temp": 36.8, "Age": 45, "Gender": 0, "ICULOS": 3}]}
```

### Low and high risk in one batch

Send the stable and the deteriorating patient together. The results come back in the same order:

```json
{
  "records": [
    {"Hour": 2, "HR": 72, "O2Sat": 98, "Temp": 36.8, "SBP": 125, "MAP": 88, "DBP": 70, "Resp": 15,
     "FiO2": 0.21, "Lactate": 1.0, "WBC": 7.5, "Creatinine": 0.9, "Platelets": 250,
     "Age": 45, "Gender": 0, "Unit1": 0, "Unit2": 1, "HospAdmTime": -2.0, "ICULOS": 3},
    {"Hour": 5, "HR": 104, "O2Sat": 94, "Temp": 38.6, "SBP": 102, "MAP": 68, "DBP": 52, "Resp": 26,
     "FiO2": 0.4, "Lactate": 3.1, "WBC": 15.2, "Creatinine": 1.4, "Platelets": 140,
     "Age": 67, "Gender": 1, "Unit1": 1, "Unit2": 0, "HospAdmTime": -12.5, "ICULOS": 6}
  ]
}
```

```json
{
  "predictions": [
    {"risk_probability": 0.0502, "prediction": 0, "warnings": []},
    {"risk_probability": 0.8584, "prediction": 1, "warnings": []}
  ],
  "threshold": 0.5,
  "model_md5": "4728f865299ee181f789f2f36b0a88ea"
}
```

From a terminal (bash or Git Bash):

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
  -d '{"records": [{"Hour": 2, "HR": 72, "Temp": 36.8, "Age": 45, "Gender": 0, "ICULOS": 3}]}'
```

From PowerShell:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/predict -Method Post -ContentType "application/json" `
  -Body '{"records": [{"Hour": 2, "HR": 72, "Temp": 36.8, "Age": 45, "Gender": 0, "ICULOS": 3}]}'
```

## Input contract

The record schema is generated from `params.yaml: validation`, the same data contract
Great Expectations enforces on the training data. Serving therefore accepts exactly
what the model was trained and validated on. Changing a bound in `params.yaml` changes
both checks at once.

| Rule | Source in `params.yaml` | Example |
|------|-------------------------|---------|
| Accepted fields: the 41 model features | `validation.ranges` keys | `HR`, `Lactate`, `ICULOS` … |
| Required fields, charted every hour | `validation.complete` | `Hour`, `Age`, `Gender`, `ICULOS` |
| Every other field is optional; omit it if not measured. The model imputes it | — | no `Lactate` this hour |
| Each value must be physically possible in the expected unit | `api.limits` | `Temp` in [10, 47] °C, `FiO2` in [0.21, 1.0] |
| Binary fields are integers 0 or 1 | `validation.binary` | `Gender`, `Unit1`, `Unit2` |
| Numbers must be JSON numbers, not strings | — | `98`, not `"98"` |
| Unknown fields are rejected | — | `hr`, `Patient_ID` |

Why reject these? Each would otherwise be scored silently as a different patient. A
misspelled `hr` would be dropped and imputed as "no heart rate measured".
`Temp: 98.6` (Fahrenheit) or `FiO2: 40` (a percentage) would reach the model as an absurd value.

### Hard limits vs. training range

Two sets of bounds per feature, both in `params.yaml`:

| Bounds | Purpose | Outside them |
|--------|---------|--------------|
| `api.limits` (wide) | What a real measurement in the expected unit can be | `422`, not scored |
| `validation.ranges` (narrower) | What the training data was checked against | Scored, with a warning |

The sickest patients are exactly the ones an early-warning model must score, so an
extreme but real value is never refused. A heart rate of 19 is still scored:

```json
{"risk_probability": 0.19, "prediction": 0,
 "warnings": ["HR=19.0 is outside the training range [20, 300]; the risk is extrapolated and less reliable"]}
```

Treat such a risk with care. The model never saw these values. Gradient boosting scores
everything beyond the training range as if it were the most extreme training value, so
HR 19 and HR 4 get the same risk. A test (`test_hard_limits_cover_every_training_value`)
guarantees the limits never reject a value the model was trained on.

## Errors

| Status | When | Body |
|--------|------|------|
| `422` | The request breaks the input contract, or the batch is empty or too large | `detail[]` with the offending field in `loc` and the reason in `msg` |
| `500` | Unexpected server error (see the logs) | — |

The server never answers without a model. If `models/model.pkl` is missing, or was
fitted on a different feature set than `params.yaml`, the server **refuses to start**.
A deployment mistake therefore shows up at startup, not as an error on every request.

## Testing it locally (Windows, macOS, Linux)

1. Get the model: `make model` (or `make repro` to train it from the data).
2. Start the server: `make api`, then open <http://127.0.0.1:8000/docs> and use **Try it out**.
3. In a second terminal, run the end-to-end check:

   ```bash
   make smoke
   ```

   ```
   [PASS] GET /health: {'status': 'ok', 'version': '0.1.0'}
   [PASS] GET /model: {'name': 'hist_gradient_boosting', 'model_md5': '...', 'threshold': 0.5}
   [PASS] POST /predict (documented example): [{'risk_probability': ..., 'prediction': ...}]
   [PASS] POST /predict (HR=0 is scored with a warning): {...}
   [PASS] POST /predict (HR beyond its hard limit is rejected): 422
   ```

From PowerShell without `make`, `Invoke-RestMethod` avoids curl's quoting issues:

```powershell
uv run uvicorn taed2_astra.api.main:app --reload            # terminal 1
Invoke-RestMethod http://127.0.0.1:8000/predict -Method Post -ContentType "application/json" `
  -Body '{"records": [{"Hour": 5, "HR": 104, "Age": 67, "Gender": 1, "ICULOS": 6}]}'   # terminal 2
```

`make serve` runs the server exactly as on the VM (no reload, one worker), which is a
good last check before deploying.

## Automated tests

`tests/test_api.py` covers every endpoint and every rejection rule above, and checks
that startup fails without a model or with one fitted on other features. These tests
run on a model fitted on synthetic in-range data, so they need neither the dataset
nor `models/model.pkl` and run in CI on every pull request:

```bash
uv run pytest tests/test_api.py
```
