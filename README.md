# Astra · Sepsis early warning for the ICU

[![CI](https://github.com/taed2-2627q1-gced-upc/taed2-astra/actions/workflows/ci.yml/badge.svg)](https://github.com/taed2-2627q1-gced-upc/taed2-astra/actions/workflows/ci.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![uv](https://img.shields.io/badge/env-uv-purple)
![DVC](https://img.shields.io/badge/data-DVC-945dd6)
![FastAPI](https://img.shields.io/badge/serving-FastAPI-009688)

Every hour, for every ICU patient, Astra estimates the risk that sepsis is developing,
from the vitals and labs charted in that hour. Trained on the PhysioNet 2019 challenge
data and served as a REST API. Built by team **Astra** for TAED2 (UPC, GCED).

> **Status:** Milestones 1–3 done (reproducible pipeline, data contract, release gates,
> CO2 reporting). Milestone 4 (API deployed on the UPC VM) is in progress.

| | |
|---|---|
| **Model** | Histogram gradient boosting · ROC-AUC 0.81 · recall 0.64 at threshold 0.5 ([model card](docs/model_card.md)) |
| **Data** | PhysioNet 2019, one row per patient-hour ([dataset card](docs/dataset_card.md)) |
| **API** | `POST /predict` scores a batch of patient-hours ([API reference](docs/api.md)) |

## Quick start

You need [uv](https://docs.astral.sh/uv/) and `make`. On Windows, install make with
`winget install ezwinports.make`. uv installs the pinned Python by itself.

```bash
git clone git@github.com:taed2-2627q1-gced-upc/taed2-astra.git
cd taed2-astra
make install          # create .venv and install the git hooks
make model            # download the trained model (needs DVC credentials, see below)
make api              # http://127.0.0.1:8000/docs
```

In a second terminal, check the running API end to end:

```bash
make smoke
```

`make model` needs a personal DagsHub token in `.dvc/config.local`. The two commands are in
[docs/development.md](docs/development.md#dagshub-credentials).

## Using the API

| Method | Path | Returns |
|--------|------|---------|
| `GET` | `/health` | Liveness and version |
| `GET` | `/model` | Served model, its MD5 (matches `dvc.lock`), threshold, features and test metrics |
| `POST` | `/predict` | One sepsis risk per patient-hour, in request order |

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"records": [{"Hour": 5, "HR": 104, "Temp": 38.6, "Age": 67, "Gender": 1, "ICULOS": 6}]}'
```

```jsonc
// illustrative values
{"predictions": [{"risk_probability": 0.31, "prediction": 0, "warnings": []}], "threshold": 0.5, "model_md5": "4728f865..."}
```

Only `Hour`, `Age`, `Gender` and `ICULOS` are required. Leave out any measurement that
was not taken. Misspelled fields, physically impossible values (a temperature of 98.6 °F,
an FiO2 of 40 %) and numbers sent as text get a `422` that names the field. Extreme but
possible values, such as an HR of 19, are scored and come back with a warning.
See [docs/api.md](docs/api.md#examples) for ready-to-send low-risk and high-risk examples and the full contract.

## Commands

| Command | What it does |
|---------|--------------|
| `make api` | Serve locally with auto-reload |
| `make serve` | Serve as in production: one worker, no reload |
| `make smoke` | Check a running API (`API_URL=http://<host>` to target another machine) |
| `make model` | Pull only the trained model from DVC |
| `make repro` | Reproduce the pipeline (`prepare → validate → train → evaluate …`) |
| `make qa` | Everything CI checks: lint, tests with coverage, notebook QA |
| `make deploy` | On the VM: pull code and model, restart, smoke-test ([guide](docs/deployment.md)) |

## Repository layout

```
├── src/taed2_astra/   # the package: data, features, modeling, api, visualization
├── tests/             # pytest suite (unit, contract, API, release gates)
├── deploy/            # systemd unit, nginx site and smoke test for the VM
├── docs/              # cards, API reference, deployment and development guides
├── notebooks/         # exploration only
├── data/  models/     # DVC-tracked, never in Git
├── metrics/  reports/ # scores, validation report, emissions, figures
├── dvc.yaml           # pipeline
└── params.yaml        # every tunable value, the data contract and release gates
```

## Documentation

| Document | For |
|----------|-----|
| [API reference](docs/api.md) | Calling the API: endpoints, input contract, errors, testing it locally |
| [Deployment guide](docs/deployment.md) | Running the API as a service on the UPC VM |
| [Development guide](docs/development.md) | Setup, remotes, pipeline, quality assurance, tooling |
| [Model card](docs/model_card.md) | What the model does, how well it scores, where it must not be used |
| [Dataset card](docs/dataset_card.md) | Where the data comes from, what is in it, ethics |
| [AGENTS.md](AGENTS.md) | Conventions: branching, commits, CI, where things go |

## Team

| Member | GitHub |
|--------|--------|
| Santiago Romagosa | [@Santi-49](https://github.com/Santi-49) |
| Pablo Fernández | [@FernanESP0](https://github.com/FernanESP0) |
| Elena Solà | [@elenasola](https://github.com/elenasola) |
| Júlia Camús | [@julietaa6](https://github.com/julietaa6) |

Licensed under [MIT](LICENSE).
