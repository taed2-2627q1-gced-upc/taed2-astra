# System design

How Astra fits together as an ML component: how a trained model gets from the pipeline
to the API, and why the design stays this simple. Calling the API is covered in
[api.md](api.md), running it on the VM in [deployment.md](deployment.md).

## Architecture

![System architecture](images/system_architecture.svg)

Astra has two halves that share one file and one contract:

- **Training** runs on a laptop or in CI. The DVC pipeline turns the PhysioNet data into
  `models/model.pkl`, which the release gates test before it is pushed to DagsHub.
- **Serving** runs on the UPC VM. It downloads that exact file, loads it once at startup
  and answers requests behind Cloudflare and nginx.
- **`params.yaml`** defines what a valid patient-hour is. The `validate` stage checks the
  training data against it, and the API builds its request schema from it.

<details>
<summary>Mermaid source</summary>

```mermaid
flowchart TB
    P[/"params.yaml<br/>schema · data contract · release gates · API limits"/]

    subgraph TRAINING["Training plane · developer laptop or CI"]
        direction LR
        PREP[prepare] --> VAL["validate<br/>Great Expectations"] --> TRAIN[train] --> EVAL[evaluate] --> CO2["co2_report<br/>plots"]
        VAL --> BENCH[benchmark]
        TRAIN --> FAIR["fairness<br/>AIF360"]
    end

    M[("models/model.pkl<br/>md5 pinned in dvc.lock")]
    GATES["Release gates<br/>tests/test_model_quality.py"]

    subgraph REMOTES["Remotes"]
        direction LR
        GH[("GitHub<br/>code · dvc.lock · CI")]
        DH[("DagsHub<br/>DVC storage · MLflow")]
    end

    subgraph SERVING["Serving plane · UPC VM (Ubuntu, 1 CPU, 2 GB)"]
        direction LR
        CFD[cloudflared] --> NGX["nginx :80"] --> UV["uvicorn 127.0.0.1:8000<br/>1 worker · systemd"] --> APP["FastAPI<br/>api/main.py · api/schemas.py"] --> PRED["modeling/predict.py"] --> MV[("models/model.pkl")]
    end

    CLIENT(["Client<br/>browser /docs · curl · script"]) -- HTTPS --> CF{{"Cloudflare<br/>astra.quick2query.com"}} -- tunnel --> SERVING

    P -. "validation rules" .-> TRAINING
    TRAINING -- "dvc repro" --> M
    GATES -. checks .-> M
    TRAINING -- "runs, metrics (MLflow)" --> DH
    M -- "dvc push" --> DH
    TRAINING -- "git push · pull request" --> GH
    GH -- "make deploy: git pull" --> SERVING
    DH -- "dvc pull models/model.pkl" --> SERVING
    P -. "same contract → Pydantic schema" .-> SERVING
```

</details>

<!-- TODO(team): confirm that cloudflared forwards to nginx on :80 (not straight to uvicorn on
:8000), and decide whether the tunnel config belongs in deploy/ next to nginx.conf. -->

## A request, step by step

![Request flow](images/request_flow.svg)

A request that breaks the input contract gets a `422` naming the field, and nothing in
the batch is scored. A valid one is scored by the model loaded at startup, and each
prediction comes back with any warnings and the `model_md5` of the model that answered.

<details>
<summary>Mermaid source</summary>

```mermaid
sequenceDiagram
    autonumber
    actor C as Client
    participant CF as Cloudflare + cloudflared
    participant N as nginx
    participant A as FastAPI (api/main.py)
    participant S as Pydantic schema (api/schemas.py)
    participant P as predict() (modeling/predict.py)

    C->>CF: POST /predict {"records": [...]}
    CF->>N: forwards, adds CF-Connecting-IP
    N->>A: proxy to 127.0.0.1:8000
    A->>S: validate every record
    alt breaks the contract (unknown field, impossible value, wrong type, batch size)
        S-->>A: errors
        A-->>C: 422, detail[].loc names the field, nothing scored
    else valid
        A->>P: records + model loaded at startup
        P-->>A: risk_probability, prediction per record
        A->>A: warnings: outside validated range, contradicting fields
        A-->>C: 200 predictions in request order + threshold + model_md5
    end
    A->>A: middleware: service log (journald) and access log (IP, 30 days)
```

</details>

## Design decisions

| Decision | Why |
|----------|-----|
| **A REST API that scores a batch per call** | A sepsis alert is needed while the clinician waits, and a ward's patients for one hour fit in one request |
| **Training and serving share only `model.pkl`** | The VM runs exactly the file the release gates tested. It never retrains and holds no patient data |
| **`model_md5` in every response** | Any prediction can be traced back to its entry in `dvc.lock`, and from there to the commit and MLflow run |
| **One contract in `params.yaml`** | Training data and live requests are checked against the same rules. Changing a bound changes both |
| **Preprocessing inside the pickled `Pipeline`** | Imputation and scaling travel with the model, so serving cannot drift from what was evaluated |
| **The server refuses to start without a valid model** | A deployment mistake shows up once, at startup, instead of as an error on every request |
| **Reject impossible values, warn about rare ones** | The sickest patients have extreme values and must still be scored. A wrong unit or a misspelled field must not be |
| **No state: no database, no patient history** | Restarts lose nothing, and the model only ever sees one patient-hour at a time |
| **No Docker, queue, feature store or model registry** | One service on one VM. Each would add moving parts without solving a problem we have; the VM-level choices are in [deployment.md](deployment.md#architecture) |
| **No authentication** | A research and teaching demo with no data behind it. A real clinical integration would need auth and rate limiting first |
