# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Runtime benchmarks for aeon's dynamic time warping
k-nearest neighbors algorithm.
"""

from __future__ import annotations

import timeit
import typing as t

import numpy as np
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
from aeon.transformations.collection import Padder
from dtaidistance import dtw_ndim
from utils import load_dataset

from sequentia.datasets.base import SequentialDataset

np.random.seed(0)
random_state: np.random.RandomState = np.random.RandomState(0)

DataSplit: t.TypeAlias = tuple[np.ndarray, np.ndarray]


def distance(s1: np.ndarray, s2: np.ndarray) -> float:
    """DTAIDistance DTW measure - not used."""
    # need to transpose sequences again
    return dtw_ndim.distance(s1.T, s2.T, use_c=True)


def prepare(data: SequentialDataset) -> DataSplit:
    """Prepare the dataset - padding."""
    # transpose sequences and pad
    X = [x.T for x, _ in data]
    padder = Padder()
    X_pad = padder.fit_transform(X)
    # X_pad = X_pad.astype("float64")
    return X_pad, data.y


def run(*, train_data: DataSplit, test_data: DataSplit, n_jobs: int) -> None:
    """Fit and predict the classifier."""
    # initialize model
    clf = KNeighborsTimeSeriesClassifier(
        n_neighbors=1,
        n_jobs=n_jobs,
        distance="dtw",
        # distance=distance,
    )

    # fit model
    X_train, y_train = train_data
    clf.fit(X_train, y_train)

    # predict model
    X_test, _ = test_data
    clf.predict(X_test)


if __name__ == "__main__":
    import argparse

    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--number", type=int, default=10)
    args: argparse.Namespace = parser.parse_args()

    train_data, test_data = load_dataset(multivariate=False)
    train_data, test_data = prepare(train_data), prepare(test_data)

    benchmark = timeit.timeit(
        "run(train_data=train_data, test_data=test_data, n_jobs=args.n_jobs)",
        globals=locals(),
        number=args.number,
    )

    print(args)  # noqa: T201
    print(f"{benchmark:.3f}s")  # noqa: T201
