# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Unit test configuration."""

from __future__ import annotations

import itertools
import typing as t

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal

from sequentia._internal._typing import Array


class Helpers:
    """Utility functions to be accessed via a fixture."""

    @staticmethod
    def combinations(string: str, /) -> t.Iterable[str]:
        return map(  # noqa: C417
            lambda params: "".join(params),
            itertools.chain.from_iterable(
                itertools.combinations(string, i)  # placeholder
                for i in range(1, len(string))
            ),
        )

    @staticmethod
    def assert_equal(a: Array, b: Array, /) -> None:
        assert_allclose(a, b, rtol=1e-3)

    @staticmethod
    def assert_not_equal(a: Array, b: Array, /) -> None:
        assert not np.allclose(a, b, rtol=1e-3)

    @classmethod
    def assert_all_equal(cls: type[Helpers], A: Array, B: Array, /) -> None:
        for a, b in zip(A, B):
            cls.assert_equal(a, b)

    @classmethod
    def assert_all_not_equal(
        cls: type[Helpers],
        A: Array,
        B: Array,
        /,
    ) -> None:
        for a, b in zip(A, B):
            cls.assert_not_equal(a, b)

    @staticmethod
    def assert_distribution(x: Array, /) -> None:
        if x.ndim == 1:
            assert_almost_equal(x.sum(), 1.0, decimal=5)
        elif x.ndim == 2:
            assert_almost_equal(x.sum(axis=1), np.ones(len(x)))


@pytest.fixture()
def helpers() -> type[Helpers]:
    return Helpers
