# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

from __future__ import annotations

import joblib

from sequentia._internal._typing import IntArray

__all__ = ["effective_n_jobs"]


def effective_n_jobs(n_jobs: int, *, x: IntArray | None = None) -> int:
    if x is None:
        return 1
    return min(joblib.effective_n_jobs(n_jobs), len(x))
