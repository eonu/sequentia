# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

import typing as t

import numpy as np

from sequentia._internal._typing import Array, IntArray

__all__ = ["get_idxs", "iter_X"]


def get_idxs(lengths: IntArray, /) -> IntArray:
    ends = lengths.cumsum()
    starts = np.zeros_like(ends)
    starts[1:] = ends[:-1]
    return np.c_[starts, ends]


def iter_X(X: Array, /, *, idxs: IntArray) -> t.Iterator[Array]:
    for start, end in idxs:
        yield X[start:end]
