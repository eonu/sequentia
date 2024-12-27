# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

from __future__ import annotations

import typing as t

import pytest

from sequentia.datasets import load_digits


@pytest.mark.parametrize("digits", [list(range(10)), [2, 5]])
def test_digits(helpers: t.Any, digits: list[int]) -> None:
    data = load_digits(digits=digits)

    assert len(data) == 300 * len(digits)
    helpers.assert_equal(data.classes, digits)
    assert set(data.y) == set(digits)

    for _, lengths, _ in data.iter_by_class():
        assert len(lengths) == 300
