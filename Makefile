.PHONY: install lint format pylint lint-nb test test-fast qa repro api clean

PYNBLINT_REPORT := reports/static_analysis/pynblint.json

install:      ## Create the environment and install the project
	uv sync
	uv run pre-commit install

format:       ## Auto-fix lint errors and formatting
	uv run ruff check --fix src tests
	uv run ruff format src tests

lint:         ## Run the static analysis CI runs (ruff + pylint)
	uv run ruff check src tests
	uv run ruff format --check src tests
	uv run pylint src tests --fail-under=9.5

pylint:       ## Pylint only, with the full report
	uv run pylint src tests

lint-nb:      ## Notebook and repository QA; fails on any lint (run `make test` first so .coverage exists)
	uv run pynblint . --yes --output $(PYNBLINT_REPORT)
	uv run python .github/scripts/check_pynblint.py $(PYNBLINT_REPORT)

test:         ## Run the whole suite; model release gates skip if models/model.pkl is absent
	uv run pytest

test-fast:    ## Run only the tests that need no data on disk (what CI's test job runs)
	uv run pytest -m "not integration"

qa: lint test lint-nb  ## Everything a pull request must pass, in CI order

repro:        ## Reproduce the DVC pipeline
	uv run dvc repro

api:          ## Serve the API locally
	uv run uvicorn taed2_astra.api.main:app --reload

clean:        ## Remove caches and build artefacts
	rm -rf .pytest_cache .coverage coverage.xml htmlcov build dist .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
