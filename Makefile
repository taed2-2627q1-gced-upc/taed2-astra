.PHONY: install lint format pylint lint-nb test repro api clean

install:      ## Create the environment and install the project
	uv sync
	uv run pre-commit install

format:       ## Auto-fix lint errors and formatting
	uv run ruff check --fix src tests
	uv run ruff format src tests

lint:         ## Run the checks CI runs
	uv run ruff check src tests
	uv run ruff format --check src tests

pylint:       ## Static analysis of the source code and tests
	uv run pylint src tests

lint-nb:      ## Notebook and repository QA (run `make test` first so .coverage exists)
	uv run pynblint notebooks
	uv run pynblint .

test:         ## Run the test suite
	uv run pytest

repro:        ## Reproduce the DVC pipeline
	uv run dvc repro

api:          ## Serve the API locally
	uv run uvicorn taed2_astra.api.main:app --reload

clean:        ## Remove caches and build artefacts
	rm -rf .pytest_cache .coverage htmlcov build dist .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
