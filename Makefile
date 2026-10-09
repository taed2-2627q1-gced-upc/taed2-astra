.PHONY: install lint format pylint lint-nb test test-fast qa repro model api serve smoke         vm-setup service-install deploy status logs clean

PYNBLINT_REPORT := reports/static_analysis/pynblint.json

# Serving. Override on the command line, e.g. `make smoke API_URL=http://<vm-ip>`.
API_APP := taed2_astra.api.main:app
HOST    ?= 127.0.0.1
PORT    ?= 8000
API_URL ?= http://$(HOST):$(PORT)
SERVICE := astra-api

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

# --- Serving: works on Windows (Git Bash or PowerShell with make) and on the VM -------------

model:        ## Download only the shipped model from the DVC remote (no data, no retraining)
	uv run dvc pull models/model.pkl

api:          ## Serve the API locally with auto-reload, for development
	uv run uvicorn $(API_APP) --host $(HOST) --port $(PORT) --reload

serve:        ## Serve the API as production does: no reload, one worker, no dev dependencies
	uv run --no-dev uvicorn $(API_APP) --host $(HOST) --port $(PORT) --workers 1

smoke:        ## Check a running API end to end (health, model, valid and rejected predictions)
	uv run --no-dev python deploy/smoke_test.py $(API_URL)

# --- VM only (Ubuntu, systemd). See docs/deployment.md -------------------------------------

vm-setup:     ## Once: install runtime dependencies and the shipped model
	uv sync --frozen --no-dev
	uv run --no-dev dvc pull models/model.pkl

service-install:  ## Once: register the API as a systemd service that starts at boot
	sed -e "s|@USER@|$$(whoami)|g" -e "s|@DIR@|$(CURDIR)|g" deploy/$(SERVICE).service 		| sudo tee /etc/systemd/system/$(SERVICE).service > /dev/null
	sudo systemctl daemon-reload
	sudo systemctl enable --now $(SERVICE)

deploy:       ## Every release: pull code and model, restart the service, smoke-test it
	git pull --ff-only
	uv sync --frozen --no-dev
	uv run --no-dev dvc pull models/model.pkl
	sudo systemctl restart $(SERVICE)
	sleep 5
	$(MAKE) smoke

status:       ## Is the service running, and since when?
	systemctl status $(SERVICE) --no-pager

logs:         ## Follow the API logs (one line per request)
	journalctl -u $(SERVICE) -f

clean:        ## Remove caches and build artefacts
	rm -rf .pytest_cache .coverage coverage.xml htmlcov build dist .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
