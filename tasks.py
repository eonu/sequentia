# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Main invoke task collection."""

from __future__ import annotations

from invoke.collection import Collection
from invoke.config import Config
from invoke.tasks import task

from make import cov, docs, lint, release, tests


@task
def install(c: Config) -> None:
    """Install package with pre-commit hooks and core, dev, docs, & test
    dependencies.
    """
    # install dependencies
    # NOTE: only including docs/tests dependencies to please editors
    c.run("poetry install --sync --only base,main,dev,docs,tests")
    # install pre-commit hooks
    c.run("pre-commit install --install-hooks")


@task
def clean(c: Config) -> None:
    """Clean temporary files, local cache and build artifacts."""
    commands: list[str] = [
        "rm -rf `find . -name __pycache__`",
        "rm -f `find . -type f -name '*.py[co]'`",
        "rm -f `find . -type f -name '*~'`",
        "rm -f `find . -type f -name '.*~'`",
        "rm -rf .cache",
        "rm -rf .pytest_cache",
        "rm -rf .ruff_cache",
        "rm -rf .tox",
        "rm -rf htmlcov",
        "rm -rf *.egg-info",
        "rm -f .coverage",
        "rm -f .coverage.*",
        "rm -rf build",
        "rm -rf dist",
        "rm -rf site",
        "rm -rf docs/build",
        "rm -rf coverage.xml",
    ]
    for command in commands:
        c.run(command)


# create top-level namespace
namespace = Collection()

# register top-level commands
for t in (install, clean):
    namespace.add_task(t)

# register namespaces
for module in (docs, tests, cov, lint, release):
    collection = Collection.from_module(module)
    namespace.add_collection(collection)
