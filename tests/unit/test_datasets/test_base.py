# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

import os
import tempfile
import typing as t

import numpy as np
import pytest
from _pytest.fixtures import SubRequest

from sequentia._internal import _data
from sequentia.datasets.base import SequentialDataset


@pytest.mark.parametrize("y_type", [int, float, None])
@pytest.mark.parametrize("use_lengths", [True, False])
def test_data(
    request: SubRequest, helpers: t.Any, y_type: type, *, use_lengths: bool
) -> None:
    X = np.atleast_2d(np.arange(10)).T

    if y_type == int:
        y = [10, 15, 10]
    elif y_type == float:
        y = [10.1, 15.2, 20.3]
    elif y_type is None:
        y = None

    if use_lengths:
        lengths = [2, 3, 5]
    else:
        lengths = None
        if y_type:
            y = y[:1]

    data = SequentialDataset(X, y, lengths=lengths)

    # X
    helpers.assert_equal(data.X, X)

    # y, classes
    if y_type == int:
        assert np.issubdtype(data.y.dtype, np.integer)
        helpers.assert_equal(data.y, np.array(y))
        helpers.assert_equal(data.classes, [10, 15] if lengths else [10])
    elif y_type == float:
        assert np.issubdtype(data.y.dtype, np.floating)
        helpers.assert_equal(data.y, np.array(y))
        assert data.classes is None
    elif y_type is None:
        for prop in ("y", "X_y", "X_y_lengths"):
            with pytest.raises(AttributeError):
                getattr(data, prop)
        assert data.classes is None

    # idxs
    if lengths:
        helpers.assert_equal(
            data.idxs,
            [
                [0, 2],
                [2, 5],
                [5, 10],
            ],
        )
    else:
        helpers.assert_equal(data.idxs, [[0, 10]])

    # _data.iter_X
    helpers.assert_equal(
        data.X,
        np.vstack(list(_data.iter_X(data.X, idxs=data.idxs))),
    )

    # __getitem__
    if y_type:
        if lengths:
            # [0]
            x, y_ = data[0]
            helpers.assert_equal(x, np.atleast_2d([0, 1]).T)
            assert y_ == y[0]
            # [:1]
            xs, ys = data[:1]
            helpers.assert_all_equal(xs, [np.atleast_2d([0, 1]).T])
            helpers.assert_equal(ys, y[:1])
            # [1:3]
            xs, ys = data[1:3]
            helpers.assert_all_equal(
                xs,
                [np.atleast_2d([2, 3, 4]).T, np.atleast_2d([5, 6, 7, 8, 9]).T],
            )
            helpers.assert_equal(ys, y[1:3])
            # [-1]
            x, y_ = data[-1]
            helpers.assert_equal(x, np.atleast_2d([5, 6, 7, 8, 9]).T)
            assert y_ == y[-1]
            # [-2:]
            xs, ys = data[-2:]
            helpers.assert_all_equal(
                xs,
                [np.atleast_2d([2, 3, 4]).T, np.atleast_2d([5, 6, 7, 8, 9]).T],
            )
            helpers.assert_equal(ys, y[-2:])
        else:
            # [0]
            x, y_ = data[0]
            helpers.assert_equal(x, X)
            assert y_ == y
            # [:1]
            xs, ys = data[:1]
            helpers.assert_all_equal(xs, [X])
            helpers.assert_equal(ys, y)
    elif lengths:
        # [0]
        x = data[0]
        helpers.assert_equal(x, np.atleast_2d([0, 1]).T)
        # [:1]
        xs = data[:1]
        helpers.assert_all_equal(xs, [np.atleast_2d([0, 1]).T])
        # [1:3]
        xs = data[1:3]
        helpers.assert_all_equal(
            xs, [np.atleast_2d([2, 3, 4]).T, np.atleast_2d([5, 6, 7, 8, 9]).T]
        )
        # [-1]
        x = data[-1]
        helpers.assert_equal(x, np.atleast_2d([5, 6, 7, 8, 9]).T)
        # [-2:]
        xs = data[-2:]
        helpers.assert_all_equal(
            xs, [np.atleast_2d([2, 3, 4]).T, np.atleast_2d([5, 6, 7, 8, 9]).T]
        )
    else:
        # [0]
        x = data[0]
        helpers.assert_equal(x, X)
        # [:1]
        xs = data[:1]
        helpers.assert_all_equal(xs, [X])

    # split
    if y and lengths:
        train, test = data.split(test_size=1 / 3, shuffle=False)
        assert len(train) == 2
        assert len(test) == 1
        helpers.assert_equal(train.lengths, data.lengths[: len(train)])
        helpers.assert_equal(test.lengths, data.lengths[-len(test) :])
        helpers.assert_equal(train.X, data.X[: train.lengths.sum()])
        helpers.assert_equal(test.X, data.X[-test.lengths.sum() :])
        helpers.assert_equal(train.y, data.y[: len(train)])
        helpers.assert_equal(test.y, data.y[-len(test) :])

    # iter_by_class
    if y_type == int and lengths:
        for X_, lengths_, c in data.iter_by_class():
            if c == 10:
                helpers.assert_equal(lengths_, [2, 5])
                helpers.assert_equal(X_, np.vstack([data.X[:2], data.X[-5:]]))
            elif c == 15:
                helpers.assert_equal(lengths_, [3])
                helpers.assert_equal(X_, data.X[2:5])

    # check serialization/deserialization
    with tempfile.TemporaryDirectory() as temp_dir:
        data_path = f"{temp_dir}/{request.node.originalname}.npz"
        # check that save works
        data.save(data_path)
        assert os.path.isfile(data_path)
        # check that load works
        data_load = SequentialDataset.load(data_path)
        # check that stored values are the same
        helpers.assert_equal(data._X, data_load._X)
        helpers.assert_equal(data._lengths, data_load._lengths)
        if y:
            helpers.assert_equal(data._y, data_load._y)
        else:
            assert data._y is None
            assert data_load._y is None
        if data._classes is not None:
            helpers.assert_equal(data._classes, data_load._classes)
        else:
            assert data._classes is None
            assert data_load._classes is None

    # copy - check that stored values are the same
    data_copy = data.copy()
    helpers.assert_equal(data._X, data_copy._X)
    assert not np.shares_memory(data._X, data_copy._X)
    helpers.assert_equal(data._lengths, data_copy._lengths)
    assert not np.shares_memory(data._lengths, data_copy._lengths)
    if y:
        helpers.assert_equal(data._y, data_copy._y)
        assert not np.shares_memory(data._y, data_copy._y)
    else:
        assert data._y is None
        assert data_copy._y is None
    if data._classes is not None:
        helpers.assert_equal(data._classes, data_copy._classes)
        assert not np.shares_memory(data._classes, data_copy._classes)
    else:
        assert data._classes is None
        assert data_copy._classes is None
