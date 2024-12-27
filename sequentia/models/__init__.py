# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Machine learning algorithms for sequence classification and regression."""

from sequentia.models.hmm import (
    CategoricalHMM,
    GaussianMixtureHMM,
    HMMClassifier,
)
from sequentia.models.knn import KNNClassifier, KNNRegressor

__all__ = [
    "CategoricalHMM",
    "GaussianMixtureHMM",
    "HMMClassifier",
    "KNNClassifier",
    "KNNRegressor",
]
