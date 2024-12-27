# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Tasks for running linting and formatting."""

from __future__ import annotations

from invoke.config import Config
from invoke.tasks import task


@task
def install(c: Config) -> None:
    """Install package with core and dev dependencies."""
    c.run("poetry install --sync --only base,main,lint")


@task
def check(c: Config) -> None:
    """Lint Python files."""
    commands: list[str] = [
        "poetry run ruff check .",
        "poetry run ruff format --check .",
        # "poetry run pydoclint .",
    ]
    for command in commands:
        c.run(command)


@task(name="format")
def format_(c: Config) -> None:
    """Format Python files."""
    commands: list[str] = [
        "poetry run ruff --fix .",
        "poetry run ruff format .",
    ]
    for command in commands:
        c.run(command)
