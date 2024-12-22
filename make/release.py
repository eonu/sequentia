# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Tasks for bumping the package version."""

import os
import re
from pathlib import Path

from invoke.config import Config
from invoke.tasks import task


@task
def build(c: Config, *, v: str) -> None:
    """Build release."""
    root: Path = Path(os.getcwd())

    # bump Sphinx documentation version - docs/source/conf.py
    conf_path: Path = root / "docs" / "source" / "conf.py"
    with open(conf_path) as f:
        conf: str = f.read()
    with open(conf_path, "w") as f:
        f.write(re.sub(r'release = ".*"', f'release = "{v}"', conf))

    # bump package version - sequentia/version.py)
    init_path: Path = root / "sequentia" / "version.py"
    with open(init_path) as f:
        init: str = f.read()
    with open(init_path, "w") as f:
        f.write(re.sub(r'VERSION = ".*"', f'VERSION = "{v}"', init))

    # bump project version - pyproject.toml
    c.run(f"poetry version -q {v}")
