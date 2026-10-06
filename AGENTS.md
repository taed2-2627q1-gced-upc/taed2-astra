# AGENTS.md

Conventions for anyone — human or AI agent — working in this repository.

## Project

`taed2-astra`: an hourly sepsis early-warning service for ICU patients, built
for the TAED2 course on the PhysioNet 2019 dataset. The DVC pipeline is
implemented end to end (`prepare → validate → benchmark / train → evaluate →
fairness, co2_report, plots`), and a FastAPI app serves the trained model. The data
contract, release gates and CO2 reporting cover Milestone 3. Milestone 4
(deployment and a fully tested API) is in progress.

## Course context

TAED2 (UPC, GCED, 2026-27 Q1) teaches software engineering for ML: each team
builds **and deploys** one ML component following MLOps best practices.
Learning objectives: apply SE and MLOps practices to build models with
**reproducibility** and **quality assurance**, then deploy them behind an API.
The repo is graded as evidence, milestone by milestone:

| Milestone | Focus | Tools | Evidence in this repo |
|-----------|-------|-------|-----------------------|
| 1 | Inception: problem, requirements, dataset and model cards | Hugging Face cards | `docs/` |
| 2 | Reproducibility: structure, versioning, experiment tracking | Cookiecutter, Git + GitHub Flow, DVC, MLflow | `dvc.yaml`, `params.yaml`, DagsHub MLflow |
| 3 | Quality assurance: CO2 reporting, static analysis, data and model tests | CodeCarbon, Pynblint, ruff + Pylint, Pytest, Great Expectations | `reports/`, `tests/`, `.github/workflows/ci.yml` |
| 4 | Deployment: ML system design and a tested, documented API | FastAPI, Pytest, cloud host | `src/taed2_astra/api/` |

Graders ask *why* a practice was used and what it caught, not only that it
exists, so prefer choices you can justify in the report.

## Setup

```bash
uv sync
cp .env.template .env   # fill in your DagsHub token
uv run pytest           # fast tests run; model release gates skip until a model exists
uv run dvc pull         # data and model, after the DVC remote setup in the README
```

Use `uv run <cmd>` for everything. Do not `pip install` into the system Python.

## Ground rules

1. **Do not invent data assumptions.** No hardcoded column names, label values,
   feature lists or thresholds anywhere in `src/`. Anything dataset-specific
   goes in `params.yaml`: `dataset` (schema), `validation` (data contract),
   `model_quality` (release gates).
2. **Keep it simple.** Prefer standard-library and scikit-learn defaults over
   custom machinery. No abstraction until there are two real call sites.
3. **One place per fact.** Paths live in `config.py`, tunables in `params.yaml`,
   dependencies in `pyproject.toml`. Do not duplicate them.
4. **Never commit data.** `data/` and `models/` are DVC-tracked. If Git asks you
   to add something there, it is a mistake.
5. **No secrets in the repo.** DagsHub tokens go in `.env` or
   `.dvc/config.local`, never in a tracked file. Tokens are personal, not
   shared team-wide. See the README for setup.
6. **A human commits and pushes.** An AI agent may write and stage changes,
   but never runs `git commit` or `git push` and never appears as author or
   co-author. See [Git](#git).

## Where things go

| Change | File |
|--------|------|
| New tunable value | `params.yaml` |
| New path | `src/taed2_astra/config.py` |
| New pipeline step | `dvc.yaml` + a module under `src/taed2_astra/` |
| New dependency | `pyproject.toml`, then `uv sync` (CI fails if `uv.lock` is stale) |
| New data rule (range, required column, prevalence) | `params.yaml: validation` |
| New model release gate | `params.yaml: model_quality` + `tests/test_model_quality.py` |
| New protected attribute or mitigation | `params.yaml: fairness` |
| New CI check | `.github/workflows/ci.yml` + `.github/rulesets/main.json` |
| Exploration | `notebooks/` — never import a notebook from `src/` |

Code that the pipeline or the API runs belongs in `src/`, not in a notebook.

## Commands

```bash
make format     # ruff: fix lint errors and format
make lint       # ruff + pylint (score >= 9.5), exactly as CI runs them
make test       # whole pytest suite, coverage gate in pyproject.toml
make test-fast  # only tests that need no data on disk (CI's "Tests" job)
make lint-nb    # Pynblint on notebooks and repo; fails on any lint
make qa         # lint + test + lint-nb: run this before opening a pull request
make repro      # dvc repro
make api        # uvicorn, docs at /docs
```

## Testing

| Layer | Where | Needs data? |
|-------|-------|-------------|
| Unit and contract tests (config, registry, metrics, energy, plots, API) | `tests/test_*.py` | No |
| Prepare stage: grouped split, no leakage, reproducible | `tests/test_make_dataset.py` | No |
| Data validation: each expectation catches its defect | `tests/test_validate.py` | No |
| Fairness audit and Reweighing: metrics, weights, serving never imports AIF360 | `tests/test_fairness.py` | No |
| Prediction contract: determinism, key order, unknown or missing fields | `tests/test_predict.py` | No |
| Release gates: performance, slices, group fairness, directional behaviour | `tests/test_model_quality.py` (`integration`) | Yes: skips without `models/model.pkl` |
| Data contract on the real splits | `validate` DVC stage (Great Expectations) | Yes |

A test's docstring says *why* the behaviour matters, not what the assert does.
New tests that need data or a trained model get `@pytest.mark.integration` and
must skip cleanly on a fresh clone.

## CI and branch protection

Every pull request into `main` runs `.github/workflows/ci.yml`. Each job is a
required status check in `.github/rulesets/main.json`:

| Check | Fails when |
|-------|-----------|
| Repository hygiene | `uv.lock` is stale, a pre-commit hook fails, or a file under `data/`/`models/` is in Git |
| Static analysis | ruff finds an error or unformatted file, or the pylint score drops below 9.5 |
| Tests | a test fails or coverage drops below the floor in `pyproject.toml` |
| Notebook QA | Pynblint reports any notebook or repository lint |
| Conventional PR title | the title is not `<type>: <description>` (it becomes the merge commit) |
| Data validation and model gates | a critical expectation fails, a release gate fails, or `dvc.lock` is stale. Runs only when the `DAGSHUB_TOKEN` secret is set |

The ruleset also requires one approving review, resolved conversations and an
up-to-date branch, and blocks force-pushes and deletion of `main`. A job's
`name` is its check's identity: rename one and update the ruleset in the same PR.

## Git

GitHub Flow: branch off `main`, open a pull request, merge after review.

- Branches: `feature/<short-description>`, `fix/<short-description>`
- Commits **and pull request titles**: [Conventional Commits](https://www.conventionalcommits.org/) —
  `<type>: <description>` in imperative present tense, e.g.
  `feat: add training stage`. Types: `feat`, `fix`, `docs`, `refactor`,
  `test`, `chore`, `ci`, `data`
- `main` is always green; the ruleset makes pushing directly to it impossible
- After `dvc repro`, run `dvc push` before opening the PR: a `dvc.lock` that
  points at artefacts missing from the remote cannot be reproduced by anyone else

### AI agents do not commit

Every commit in this repository is authored and pushed by a team member. An
AI assistant may edit and stage files, but the human reviews the diff and
presses the button.

- No `Co-Authored-By:` trailer naming an AI, and no AI in the author field
- No `Generated with ...` footers in commit messages or pull request bodies
- The agent does not run `git commit`, `git push`, `git merge` or `gh pr create`

The point is accountability: the person whose name is on the commit is the
person who read the diff and vouches for it. This is a course deliverable
assessed on our work, and Git history is part of the evidence.

The full commit format lives in
[.github/copilot-commit-message-instructions.md](.github/copilot-commit-message-instructions.md).
VS Code's AI commit-message generator reads that same file, so generated and
hand-written messages follow one convention.

## Style

- Formatting is `ruff format` at 120 columns — do not hand-format; run `make format`
- Public functions get a one-line docstring saying what they return
- Comments explain *why*, never *what* the line already says
- Type hints on function signatures
