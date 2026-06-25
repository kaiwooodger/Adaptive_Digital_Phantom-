PYTHON ?= python3
VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip

.PHONY: setup setup-medical smoke test check clean

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PIP) install -e ".[dev]"

setup-medical:
	$(PYTHON) -m venv $(VENV)
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PIP) install -e ".[dev,medical]"

smoke:
	./scripts/smoke_test_cli.sh $(VENV_PYTHON)

test:
	$(VENV_PYTHON) -m pytest -q

check: smoke test

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
	find src tests -type d -name __pycache__ -prune -exec rm -rf {} +
