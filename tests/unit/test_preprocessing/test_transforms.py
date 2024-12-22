# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

import typing as t

import numpy as np
import pytest
from _pytest.fixtures import SubRequest
from sklearn.preprocessing import minmax_scale

from sequentia._internal import _data
from sequentia._internal._typing import Array
from sequentia.datasets import SequentialDataset, load_digits
from sequentia.preprocessing import transforms

from ...conftest import Helpers


@pytest.fixture(scope="module")
def random_state(request: SubRequest) -> np.random.RandomState:
    return np.random.RandomState(1)


@pytest.fixture(scope="module")
def data(random_state: np.random.RandomState) -> SequentialDataset:
    data_ = load_digits(digits=[0])
    _, subset = data_.split(
        test_size=0.2, random_state=random_state, stratify=True
    )
    return subset


def check_filter(x: Array, xt: Array, func: t.Callable, k: int) -> None:
    """NOTE: Only works for odd k"""
    assert len(x) == len(xt)
    Helpers.assert_equal(xt[k // 2], func(x[:k], axis=0))


def test_function_transformer(helpers: t.Any, data: SequentialDataset) -> None:
    # create the transform
    transform = transforms.IndependentFunctionTransformer(minmax_scale)
    # check that fit works - should do nothing
    transform.fit(**data.X_lengths)
    # check that fit_transform works - shouldn't do anything on fit, but should transform
    X_fit_transform = transform.fit_transform(**data.X_lengths)
    # check that transform works
    X_transform = transform.transform(**data.X_lengths)
    # check that fit_transform and transform produce the same transformed data
    helpers.assert_equal(X_fit_transform, X_transform)
    # check that features of each sequence are independently scaled to [0, 1]
    for xt in _data.iter_X(X_transform, idxs=data.idxs):
        helpers.assert_equal(xt.min(axis=0), np.zeros(xt.shape[1]))
        helpers.assert_equal(xt.max(axis=0), np.ones(xt.shape[1]))


@pytest.mark.parametrize("avg", ["mean", "median"])
@pytest.mark.parametrize("k", [3, 5])
def test_filters(
    data: SequentialDataset,
    random_state: np.random.RandomState,
    avg: t.Literal["mean", "median"],
    k: int,
) -> None:
    filter_ = getattr(transforms, f"{avg}_filter")
    check_filter_ = lambda x, xt: check_filter(x, xt, getattr(np, avg), k)

    # check that filters are correctly applied for a single sequence
    n_features = 2
    x = random_state.rand(10 * n_features).reshape(-1, n_features)
    xt = filter_(x, k=k)
    check_filter_(x, xt)

    # create a transform using the filter, passing k
    transform = transforms.IndependentFunctionTransformer(
        filter_, kw_args={"k": k}
    )
    Xt = transform.transform(**data.X_lengths)

    # check that filters are correctly applied for multiple sequences
    idxs = _data.get_idxs(data.lengths)
    for x, xt in zip(
        *map(lambda X: _data.iter_X(X, idxs=idxs), (data.X, Xt))  # noqa: C417
    ):
        check_filter_(x, xt)
