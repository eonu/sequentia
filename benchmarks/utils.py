# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Utilities for benchmarking."""

from __future__ import annotations

import numpy as np

from sequentia.datasets.base import SequentialDataset
from sequentia.datasets.digits import load_digits

__all__ = ["load_dataset"]

np.random.seed(0)
random_state: np.random.RandomState = np.random.RandomState(0)


def load_dataset(
    *, multivariate: bool
) -> tuple[SequentialDataset, SequentialDataset]:
    """Loads the Free Spoken Digit Dataset."""
    # load data
    data: SequentialDataset = load_digits()

    # split dataset
    train_data, test_data = data.split(
        test_size=0.5,
        random_state=random_state,
        shuffle=True,
        stratify=True,
    )

    if multivariate:
        # return untransformed data
        return train_data, test_data

    # retrieve features
    X_train, X_test = train_data.X, test_data.X

    # reduce to one dimension
    X_train = X_train.mean(axis=-1, keepdims=True)
    X_test = X_test.mean(axis=-1, keepdims=True)

    # return splits
    train_split: SequentialDataset = SequentialDataset(
        X=X_train,
        y=train_data.y,
        lengths=train_data.lengths,
        classes=train_data.classes,
    )
    test_split: SequentialDataset = SequentialDataset(
        X=X_test,
        y=test_data.y,
        lengths=test_data.lengths,
        classes=test_data.classes,
    )
    return train_split, test_split
