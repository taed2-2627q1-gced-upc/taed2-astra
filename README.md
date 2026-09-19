# taed2-astra

Patient risk prediction service — TAED2 course project, team **Astra**.

> **Status: scaffold.** The structure, tooling and pipeline wiring are in place.
> The modelling logic is deliberately left unimplemented — every stub is marked
> with `TODO(team)` and raises `NotImplementedError`.

## Team

| Member | GitHub |
|--------|--------|
| Santiago Romagosa | [@Santi-49](https://github.com/Santi-49) |
| Pablo Fernández | [@FernanESP0](https://github.com/FernanESP0) |
| Elena Solà | [@elenasola](https://github.com/elenasola) |
| Júlia Camús | [@julietaa6](https://github.com/julietaa6) |

## Getting started

```bash
uv sync                  # create .venv and install everything
uv run pre-commit install
cp .env.template .env    # then fill in your DagsHub token
uv run pytest            # should pass on a fresh clone
```

`uv` installs the Python version pinned in `.python-version`, so no separate
interpreter setup is needed. Install it from <https://docs.astral.sh/uv/>.

> On Windows, the course expects you to work inside
> [WSL](https://learn.microsoft.com/en-us/windows/wsl/install).
> Clone over **SSH**, not HTTPS:
> `git clone git@github.com:taed2-2627q1-gced-upc/taed2-astra.git`

Open the folder in VS Code and accept the recommended extensions when prompted.
`.vscode/settings.json` is tracked, so formatting, linting and AI-generated
commit messages behave identically for everyone.

## Repository layout

```
├── data/              # DVC-tracked, never committed to Git
│   ├── raw/           # immutable input, exactly as downloaded
│   └── processed/     # model-ready train/test splits
├── docs/              # dataset card, model card
├── metrics/           # metrics.json, committed so PR diffs show score changes
├── models/            # DVC-tracked trained artefacts
├── notebooks/         # exploration only; production code lives in src/
├── reports/           # validation report, CodeCarbon emissions, figures
├── src/taed2_astra/   # the installable package
│   ├── config.py      # paths and params — the only place that knows the layout
│   ├── data/          # make_dataset.py (prepare), validate.py (expectations)
│   ├── features/      # transformations shared by training and serving
│   ├── modeling/      # train.py, evaluate.py, predict.py
│   ├── api/           # FastAPI app and request/response schemas
│   └── visualization/ # plots for the report and the model card
├── tests/             # pytest suite
├── dvc.yaml           # pipeline definition
└── params.yaml        # every tunable value
```

This follows `cookiecutter-data-science` with three deliberate deviations:

1. **`src/` layout** instead of a top-level package folder, so the project is
   installed rather than imported by path. Imports then behave identically in
   tests, in DVC stages and in the deployed API.
2. **`api/` package added**, since the course requires a FastAPI deployment,
   which the template does not cover.
3. **Unused template folders removed** (`data/interim/`, `data/external/`,
   `references/`). Add one back the day it holds something — an empty folder
   is noise in every `ls` and every review.

## Remotes

Three separate remotes, each owning one kind of artefact. The code repository
belongs to the course organisation; the DagsHub project is owned by a team
member, carries the **same name**, and only stores data and experiment runs.
All four of us must be collaborators on it.

| What | Where | Command |
|------|-------|---------|
| Code | GitHub `taed2-2627q1-gced-upc/taed2-astra` | `git push` |
| Data and models | DagsHub S3 storage | `dvc push` / `dvc pull` |
| Experiments | DagsHub MLflow server | automatic during `dvc repro` |

### One-time setup per clone

Get a token from DagsHub (**Settings → Tokens**), then:

```bash
# Data remote - writes .dvc/config.local, which is gitignored.
# DagsHub issues one token, not a key/secret pair: use it for BOTH fields.
dvc remote modify --local dagshub access_key_id     <token>
dvc remote modify --local dagshub secret_access_key <token>

# Experiment tracking - copy the template and fill it in
cp .env.template .env
```

`.env` is gitignored and loaded automatically. Never commit a token; tokens are
personal, not shared team-wide. The DagsHub URLs in `.dvc/config` and
`params.yaml` are public and safe to track.

To log runs locally instead of to DagsHub, set `MLFLOW_TRACKING_URI=file:./mlruns`
in `.env` — it overrides `params.yaml` without editing a DVC-tracked file.

## Workflow

**Pipeline** — `dvc repro` runs `prepare → validate → {benchmark, train → evaluate}`,
skipping any stage whose dependencies and params are unchanged.

**Model selection** — `benchmark` scores every candidate listed in
`params.yaml` (`benchmark.models`) on the same patient-grouped CV folds and
writes the ranking to `metrics/benchmark.json`. `train.model` picks the
candidate that ships. Promote one with `dvc exp run -S train.model=<name>`;
iterate quickly with `-S benchmark.max_rows=200000`. The `tabpfn` candidate
(Hugging Face `Prior-Labs/TabPFN-v2-clf`) needs `uv sync --group foundation`
and is skipped otherwise.

**Experiments** — browse runs on the DagsHub MLflow tab. The benchmark logs one
nested run per candidate (params, mean/std metrics, latency, model size,
emissions, model artifact); training logs the shipped model the same way.

**API** — `make api`, then open <http://127.0.0.1:8000/docs>.

**Data** — `dvc pull` / `dvc push`. Never `git add` anything under `data/` or
`models/`.

See [AGENTS.md](AGENTS.md) for branching, commit and review conventions, and
[docs/](docs/) for the dataset and model cards.

## Tooling

| Concern | Tool |
|---------|------|
| Code versioning | Git + GitHub Flow |
| Data versioning | DVC |
| Experiment tracking | MLflow |
| Data validation | Great Expectations |
| Testing | Pytest |
| Linting and formatting | Ruff |
| Notebook quality | Pynblint |
| Sustainability | CodeCarbon |
| Serving | FastAPI |
