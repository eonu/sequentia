# This Makefile is based on https://github.com/pydantic/pydantic/blob/main/Makefile.

.DEFAULT_GOAL := dev

# check that poetry is installed
.PHONY: .check-poetry
.check-poetry:
	@poetry -V || echo 'Please install Poetry: https://python-poetry.org/'

# install invoke and tox
.PHONY: base
base: .check-poetry
	poetry install --sync --only base

# install development dependencies
.PHONY: dev
dev: .check-poetry
	poetry install --sync --only base
	poetry run invoke install
