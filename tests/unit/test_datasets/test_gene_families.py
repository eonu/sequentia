# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

from __future__ import annotations

import typing as t

import pytest

from sequentia.datasets import load_gene_families

counts = {0: 531, 1: 534, 2: 349, 3: 672, 4: 711, 5: 240, 6: 1343}


@pytest.mark.parametrize("families", [list(range(7)), [2, 5]])
def test_gene_families(helpers: t.Any, families: list[int]) -> None:
    data, enc = load_gene_families(families=families)

    assert set(enc.classes_) == {"A", "C", "G", "N", "T"}

    helpers.assert_equal(data.classes, families)
    assert set(data.y) == set(families)

    for family in families:
        assert (data.y == family).sum() == counts[family]
