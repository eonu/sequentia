# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Runtime benchmarks for sktime's dynamic time warping
k-nearest neighbors algorithm.
"""

from __future__ import annotations

import timeit
import typing as t

import numpy as np
import pandas as pd
from dtaidistance import dtw_ndim
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from utils import load_dataset

from sequentia.datasets.base import SequentialDataset

np.random.seed(0)
random_state: np.random.RandomState = np.random.RandomState(0)

DataSplit: t.TypeAlias = tuple[pd.Series, np.ndarray]


def distance(s1: pd.Series, s2: pd.Series) -> np.ndarray:
    """DTAIDistance DTW measure - not used."""
    s1, s2 = s1.droplevel(1), s2.droplevel(1)
    m = s1.index.max() + 1
    n = s2.index.max() + 1
    matrix = np.zeros((m, n))
    for i in range(m):
        a = np.trim_zeros(s1.loc[i].to_numpy(dtype=np.float64))
        for j in range(n):
            b = np.trim_zeros(s2.loc[j].to_numpy(dtype=np.float64))
            matrix[i][j] = dtw_ndim.distance(a, b, use_c=True)
    return matrix


def pad(x: np.ndarray, length: int) -> np.ndarray:
    """Pad a sequence with zeros."""
    return np.concat((x, np.zeros((length - len(x), x.shape[-1]))))


def prepare(data: SequentialDataset) -> DataSplit:
    """Prepare the dataset - pad and convert to multi-indexed
    Pandas DataFrame.
    """
    # convert to padded pandas multi-index
    length = data.lengths.max()
    X = [pd.DataFrame(pad(x, length=length)) for x, _ in data]
    X_pd = pd.concat(X, keys=range(len(X)), axis=0)
    return X_pd, data.y


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
