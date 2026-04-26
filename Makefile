.PHONY: check lint test format install-dev

PYTHON ?= python

install-dev:
	$(PYTHON) -m pip install -e ".[dev]"

check:
	$(PYTHON) -m compileall src

test:
	$(PYTHON) -m unittest discover -s tests

lint:
	$(PYTHON) -m ruff check src tests

format:
	$(PYTHON) -m ruff format src tests
