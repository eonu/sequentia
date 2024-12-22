# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Adds a notice to the top of all Python source code files.

This script is based on:
https://github.com/fatiando/maintenance/issues/10#issuecomment-718754908
"""

from pathlib import Path

notice = """
# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).
""".strip()


for f in Path(".").glob("**/*.py"):
    if not str(f).startswith("."):
        code = f.read_text()
        if not code.startswith(notice):
            f.write_text(f"{notice}\n\n{code}")
