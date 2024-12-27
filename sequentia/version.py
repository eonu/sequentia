# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Version information for Sequentia.

Source code modified from pydantic (https://github.com/pydantic/pydantic).

    The MIT License (MIT)

    Copyright (c) 2017 to present Pydantic Services Inc. and individual
    contributors.

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.
"""

__all__ = ["VERSION", "version_info"]

VERSION = "2.5.0"


def version_info() -> str:
    """Return complete version information for Sequentia and its
    dependencies.
    """
    import importlib.metadata
    import importlib.util
    import platform
    import sys
    from pathlib import Path

    # get data about packages that:
    # - are closely related to Sequentia,
    # - use Sequentia,
    # - often conflict with Sequentia.
    package_names = {
        "numba",
        "numpy",
        "hmmlearn",
        "dtaidistance",
        "scikit-learn",
        "scipy",
        "joblib",
        "pydantic",
    }
    related_packages = []

    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"]
        if name in package_names:
            entry = f"{name}-{dist.version}"
            if name == "dtaidistance":
                clib = bool(importlib.util.find_spec("dtaidistance.dtw_cc"))
                entry = f"{entry} (c={clib})"
            related_packages.append(entry)

    info = {
        "sequentia version": VERSION,
        "install path": Path(__file__).resolve().parent,
        "python version": sys.version,
        "platform": platform.platform(),
        "related packages": ", ".join(related_packages),
    }
    return "\n".join(
        "{:>30} {}".format(k + ":", str(v).replace("\n", " "))  #
        for k, v in info.items()
    )
