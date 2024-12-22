# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Tasks for running coverage checks."""

from invoke.config import Config
from invoke.tasks import task


@task
def install(c: Config) -> None:
    """Install package with core and coverage dependencies."""
    c.run("poetry install --sync --only base,main,cov")
