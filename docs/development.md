# Development guide

Everything a team member needs to work on the project: setup, remotes, the pipeline,
quality assurance and the tooling behind each. Conventions for branches, commits and
reviews live in [AGENTS.md](../AGENTS.md).

## Setup

```bash
git clone git@github.com:taed2-2627q1-gced-upc/taed2-astra.git   # SSH, not HTTPS
cd taed2-astra
make install             # uv sync + pre-commit hooks
cp .env.template .env    # then fill in your DagsHub token
uv run pytest            # passes on a fresh clone; release gates skip without a model
```

[uv](https://docs.astral.sh/uv/) installs the Python version pinned in `.python-version`,
so no separate interpreter setup is needed. On Windows the project works natively
(PowerShell or Git Bash, with `winget install ezwinports.make`) and in
[WSL](https://learn.microsoft.com/en-us/windows/wsl/install).

Open the folder in VS Code and accept the recommended extensions. `.vscode/settings.json`
is tracked, so formatting, linting and AI-generated commit messages behave identically
for everyone.

## Remotes

Three remotes, each owning one kind of artefact. The code repository belongs to the
course organisation. The DagsHub project is owned by a team member, has the **same
name**, and only stores data and experiment runs. All four of us are collaborators on it.

| What | Where | Command |
|------|-------|---------|
| Code | GitHub `taed2-2627q1-gced-upc/taed2-astra` | `git push` |
| Data and models | DagsHub S3 storage | `dvc push` / `dvc pull` |
| Experiments | DagsHub MLflow server | automatic during `dvc repro` |

### DagsHub credentials

Get a token from DagsHub (**Settings → Tokens**), then:

```bash
# Data remote: writes .dvc/config.local, which is gitignored.
# DagsHub issues one token, not a key/secret pair: use it for BOTH fields.
uv run dvc remote modify --local dagshub access_key_id     <token>
uv run dvc remote modify --local dagshub secret_access_key <token>

# Experiment tracking: copy the template and fill it in
cp .env.template .env
```

`.env` is gitignored and loaded automatically. Never commit a token. Tokens are
personal, not shared team-wide. The DagsHub URLs in `.dvc/config` and `params.yaml`
are public and safe to track.

To log runs locally instead of to DagsHub, set `MLFLOW_TRACKING_URI=sqlite:///mlflow.db`
in `.env`. It overrides `params.yaml` without editing a DVC-tracked file. Browse the
runs with `uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`.

## Pipeline

`make repro` (`dvc repro`) runs
`prepare → validate → {benchmark, train → evaluate → co2_report}` and `plots`,
skipping any stage whose dependencies and params are unchanged.

| Stage | Produces |
|-------|----------|
| `prepare` | Patient-grouped train/test split in `data/processed/` |
| `validate` | Great Expectations data contract on both splits → `reports/data_validation.json` |
| `benchmark` | Every candidate on the same grouped CV folds → `metrics/benchmark.json` |
| `train` | The shipped model → `models/model.pkl` |
| `evaluate` | Test-split metrics → `metrics/metrics.json` |
| `co2_report`, `plots` | Emissions in the model card, figures in `reports/figures/` |

**After a repro, run `dvc push` before opening a pull request.** Otherwise `dvc.lock`
points at artefacts nobody else (CI, teammates, the VM) can pull.

**Model selection.** `benchmark` ranks the candidates in `benchmark.models`.
`train.model` picks the one that ships. Promote another with
`dvc exp run -S train.model=<name>`, and iterate quickly with `-S benchmark.max_rows=200000`.
The `tabpfn` candidate (Hugging Face `Prior-Labs/TabPFN-v2-clf`) needs
`uv sync --group foundation` and is skipped otherwise.

**Experiments.** Browse runs on the DagsHub MLflow tab. The benchmark logs one nested
run per candidate (params, mean/std metrics, latency, model size, emissions, model
artifact). Training logs the shipped model the same way.

**Sustainability figures.** `uv run astra-plots` redraws the energy figures in
`reports/figures/` from `reports/emissions/emissions.csv`.

## Quality assurance

`make qa` runs what CI runs: ruff and pylint (score ≥ 9.5), the pytest suite with a
coverage floor, and Pynblint.

| Check | Where | Gate |
|-------|-------|------|
| Data contract (ranges, completeness, label rate, dtypes) | `validate` stage, `params.yaml: validation` | Critical expectations stop the pipeline |
| Model release gates (ROC-AUC, PR-AUC lift, recall, slices, directional) | `tests/test_model_quality.py`, `params.yaml: model_quality` | A failing gate blocks the merge |
| Prediction and API contract | `tests/test_predict.py`, `tests/test_api.py` | Run without data on every pull request |

See [AGENTS.md](../AGENTS.md#ci-and-branch-protection) for the CI jobs that protect `main`.

## Project structure

```
├── data/              # DVC-tracked, never committed to Git
│   ├── raw/           # immutable input, exactly as downloaded
│   └── processed/     # model-ready train/test splits
├── deploy/            # systemd unit and smoke test (see deployment.md)
├── docs/              # cards and guides
├── metrics/           # metrics.json, committed so PR diffs show score changes
├── models/            # DVC-tracked trained artefacts
├── notebooks/         # exploration only; production code lives in src/
├── reports/           # validation report, CodeCarbon emissions, figures
├── src/taed2_astra/   # the installable package
│   ├── config.py      # paths and params: the only place that knows the layout
│   ├── data/          # make_dataset.py (prepare), validate.py (expectations)
│   ├── features/      # transformations shared by training and serving
│   ├── modeling/      # registry, benchmark, train, evaluate, predict
│   ├── api/           # FastAPI app and request/response schemas
│   └── visualization/ # plots for the report and the model card
├── tests/             # pytest suite
├── dvc.yaml           # pipeline definition
└── params.yaml        # every tunable value
```

This follows `cookiecutter-data-science` with deliberate deviations:

1. **`src/` layout** instead of a top-level package folder, so the project is installed
   rather than imported by path. Imports then behave identically in tests, in DVC
   stages and in the deployed API.
2. **`api/` and `deploy/` added**, since the course requires a deployed FastAPI service,
   which the template does not cover.
3. **Unused template folders removed** (`data/interim/`, `data/external/`, `references/`).
   Add one back the day it holds something.

## Tooling

| Concern | Tool |
|---------|------|
| Environment and dependencies | uv (`pyproject.toml` + `uv.lock`) |
| Code versioning | Git + GitHub Flow |
| Data and model versioning | DVC (DagsHub remote) |
| Experiment tracking | MLflow (DagsHub) |
| Data validation | Great Expectations |
| Testing | Pytest + pytest-cov |
| Linting and formatting | Ruff, Pylint |
| Notebook and repository quality | Pynblint |
| Sustainability | CodeCarbon |
| CI and branch protection | GitHub Actions + repository ruleset |
| Serving | FastAPI + uvicorn, managed by systemd, behind a Cloudflare tunnel |
