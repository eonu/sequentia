# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Tasks for running tests."""

from __future__ import annotations

from invoke.config import Config
from invoke.tasks import task


@task
def install(c: Config) -> None:
    """Install package with core and test dependencies."""
    c.run("poetry install --sync --only base,main,tests")


@task
def unit(c: Config, *, cov: bool = False) -> None:
    """Run unit tests."""
    command: str = "poetry run pytest tests/"

    if cov:
        command = (
            f"{command} --cov-config .coveragerc "
            "--cov sequentia --cov-report xml"
        )
    c.run(command)
