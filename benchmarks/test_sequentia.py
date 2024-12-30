# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Runtime benchmarks for sequentia's dynamic time warping
k-nearest neighbors algorithm.
"""

from __future__ import annotations

import timeit

import numpy as np
from utils import load_dataset

import sequentia
from sequentia.datasets.base import SequentialDataset

np.random.seed(0)
random_state: np.random.RandomState = np.random.RandomState(0)


def run(
    *, train_data: SequentialDataset, test_data: SequentialDataset, n_jobs: int
) -> None:
    """Fit and predict the classifier."""
    # initialize model
    clf = sequentia.models.KNNClassifier(
        k=1,
        use_c=True,
        n_jobs=n_jobs,
        random_state=random_state,
        classes=train_data.classes,
    )

    # fit model
    clf.fit(X=train_data.X, y=train_data.y, lengths=train_data.lengths)

    # predict model
    clf.predict(X=test_data.X, lengths=test_data.lengths)


if __name__ == "__main__":
    import argparse

    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--number", type=int, default=10)
    args: argparse.Namespace = parser.parse_args()

    train_data, test_data = load_dataset(multivariate=False)

    benchmark = timeit.timeit(
        "run(train_data=train_data, test_data=test_data, n_jobs=args.n_jobs)",
        globals=locals(),
        number=args.number,
    )

    print(args)  # noqa: T201
    print(f"{benchmark:.3f}s")  # noqa: T201
