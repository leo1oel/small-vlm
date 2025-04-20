# Makefile for easy development workflows.
# See development.md for docs.
# Note GitHub Actions call uv directly, not this Makefile.

.DEFAULT_GOAL := default

.PHONY: default install lint test upgrade build clean

default: install lint test

install:
	uv sync --all-extras --dev
	uv pip install flash-attn --no-build-isolation

lint:
	uv run python devtools/lint.py

test:
	uv run pytest -vv -s

upgrade:
	uv sync --upgrade
	uv pip install flash-attn --no-build-isolation

build:
	uv build

clean:
	-rm -rf dist/
	-rm -rf *.egg-info/
	-rm -rf .pytest_cache/
	-rm -rf .mypy_cache/
	-rm -rf .venv/
	-find . -type d -name "__pycache__" -exec rm -rf {} +