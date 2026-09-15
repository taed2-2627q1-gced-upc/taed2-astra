# AGENTS.md

Conventions for anyone — human or AI agent — working in this repository.

## Project

`taed2-astra`: a patient risk prediction service built for the TAED2 course.
The repository is currently a **scaffold**: structure and tooling are wired,
modelling logic is not. Stubs raise `NotImplementedError` and are marked
`TODO(team)`.

## Setup

```bash
uv sync
cp .env.template .env   # fill in your DagsHub token
uv run pytest
```

Use `uv run <cmd>` for everything. Do not `pip install` into the system Python.

## Ground rules

1. **Do not invent data assumptions.** The dataset is not fixed yet. No
   hardcoded column names, label values or feature lists anywhere in `src/`.
   Anything dataset-specific goes in `params.yaml`.
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
| New dependency | `pyproject.toml`, then `uv sync` |
| Exploration | `notebooks/` — never import a notebook from `src/` |

Code that the pipeline or the API runs belongs in `src/`, not in a notebook.

## Commands

```bash
make format   # ruff: fix lint errors and format
make lint     # ruff: the checks CI runs
make test     # pytest
make repro    # dvc repro
make api      # uvicorn, docs at /docs
```

Run `make lint` and `make test` before opening a pull request; CI runs the same.

## Git

GitHub Flow: branch off `main`, open a pull request, merge after review.

- Branches: `feature/<short-description>`, `fix/<short-description>`
- Commits: imperative present tense — `add training stage`, not `added`
- `main` is always green; never push directly to it

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
