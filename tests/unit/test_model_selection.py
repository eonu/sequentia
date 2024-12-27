# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest
from sklearn.model_selection._split import (
    BaseCrossValidator,
    BaseShuffleSplit,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import minmax_scale

from sequentia.datasets import SequentialDataset, load_digits
from sequentia.enums import CovarianceMode, PriorMode, TopologyMode
from sequentia.model_selection import (
    GridSearchCV,
    HalvingGridSearchCV,
    KFold,
    RandomizedSearchCV,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    ShuffleSplit,
    StratifiedKFold,
    StratifiedShuffleSplit,
    param_grid,
)
from sequentia.model_selection._search import BaseSearchCV
from sequentia.models import (
    GaussianMixtureHMM,
    HMMClassifier,
    KNNClassifier,
    KNNRegressor,
)
from sequentia.preprocessing import IndependentFunctionTransformer

EPS: np.float32 = np.finfo(np.float32).eps
random_state: np.random.RandomState = np.random.RandomState(0)


def exp_weight(x: np.ndarray) -> np.ndarray:
    return np.exp(-x)


def inv_weight(x: np.ndarray) -> np.ndarray:
    return 1 / (x + EPS)


@pytest.fixture(scope="module")
def data() -> SequentialDataset:
    """Small subset of the spoken digits dataset."""
    digits = load_digits(digits={0, 1})
    _, digits = digits.split(
        test_size=0.1,
        random_state=random_state,
        shuffle=True,
        stratify=True,
    )
    return digits


@pytest.mark.parametrize(
    "cv",
    [
        KFold,
        StratifiedKFold,
        ShuffleSplit,
        StratifiedShuffleSplit,
        RepeatedKFold,
        RepeatedStratifiedKFold,
    ],
)
@pytest.mark.parametrize(
    "search", [GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV]
)
def test_knn_classifier(
    data: SequentialDataset,
    search: type[BaseSearchCV],
    cv: type[BaseCrossValidator] | type[BaseShuffleSplit],
) -> None:
    # Specify cross-validator parameters
    cv_kwargs = {"random_state": 0, "n_splits": 2}
    if cv in (KFold, StratifiedKFold):
        cv_kwargs["shuffle"] = True

    # Initialize search, splitter and parameter
    optimizer = search(
        Pipeline(
            [
                ("scale", IndependentFunctionTransformer(minmax_scale)),
                ("knn", KNNClassifier(use_c=True, n_jobs=-1)),
            ]
        ),
        {
            "knn__k": [1, 5],
            "knn__weighting": [exp_weight, inv_weight],
        },
        cv=cv(**cv_kwargs),
        n_jobs=-1,
    )

    # Perform the hyper-parameter search and retrieve the best model
    optimizer.fit(data.X, data.y, lengths=data.lengths)
    assert optimizer.best_score_ > 0.8
    clf = optimizer.best_estimator_

    # Predict labels
    y_pred = clf.predict(data.X, lengths=data.lengths)
    assert np.isin(y_pred, (0, 1)).all()

    # Predict probabilities
    y_probs = clf.predict_proba(data.X, lengths=data.lengths)
    assert ((y_probs >= 0) & (y_probs <= 1)).all()
    npt.assert_almost_equal(y_probs.sum(axis=1), 1.0)

    # Predict log probabilities
    y_log_probs = clf.predict_log_proba(data.X, lengths=data.lengths)
    assert (y_log_probs <= 0).all()
    npt.assert_almost_equal(y_log_probs, np.log(y_probs))

    # Calculate accuracy
    acc = clf.score(data.X, data.y, lengths=data.lengths)
    assert acc > 0.8


@pytest.mark.parametrize(
    "cv",
    [
        KFold,
        StratifiedKFold,
        ShuffleSplit,
        StratifiedShuffleSplit,
        RepeatedKFold,
        RepeatedStratifiedKFold,
    ],
)
@pytest.mark.parametrize(
    "search", [GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV]
)
def test_knn_regressor(
    data: SequentialDataset,
    search: type[BaseSearchCV],
    cv: type[BaseCrossValidator] | type[BaseShuffleSplit],
) -> None:
    # Specify cross-validator parameters
    cv_kwargs = {"random_state": 0, "n_splits": 2}
    if cv in (KFold, StratifiedKFold):
        cv_kwargs["shuffle"] = True

    # Initialize search, splitter and parameter
    optimizer = search(
        Pipeline(
            [
                ("scale", IndependentFunctionTransformer(minmax_scale)),
                ("knn", KNNRegressor(use_c=True, n_jobs=-1)),
            ]
        ),
        {
            "knn__k": [3, 5],
            "knn__weighting": [exp_weight, inv_weight],
        },
        cv=cv(**cv_kwargs),
        n_jobs=-1,
    )

    # Convert labels to float
    y = data.y.astype(np.float64)

    # Perform the hyper-parameter search and retrieve the best model
    optimizer.fit(data.X, y, lengths=data.lengths)
    assert optimizer.best_score_ > 0.8
    model = optimizer.best_estimator_

    # Predict labels
    y_pred = model.predict(data.X, lengths=data.lengths)
    assert ((y_pred >= 0) & (y_pred <= 1)).all()

    # Calculate R^2
    r2 = model.score(data.X, y, lengths=data.lengths)
    assert r2 > 0.8


def test_hmm_classifier(data: SequentialDataset) -> None:
    # Initialize search, splitter and parameter
    optimizer = GridSearchCV(
        estimator=Pipeline(
            [
                ("scale", IndependentFunctionTransformer(minmax_scale)),
                ("clf", HMMClassifier(variant=GaussianMixtureHMM, n_jobs=-1)),
            ]
        ),
        param_grid={
            "clf__prior": [PriorMode.UNIFORM, PriorMode.FREQUENCY],
            "clf__model_kwargs": param_grid(
                n_states=[3, 4, 5],
                n_components=[2, 3, 4],
                covariance=[CovarianceMode.DIAGONAL, CovarianceMode.SPHERICAL],
                topology=[TopologyMode.LEFT_RIGHT, TopologyMode.LINEAR],
            ),
        },
        cv=StratifiedKFold(),
        n_jobs=-1,
    )

    # Perform the hyper-parameter search and retrieve the best model
    optimizer.fit(data.X, data.y, lengths=data.lengths)
    assert optimizer.best_score_ > 0.8
    clf = optimizer.best_estimator_

    # Predict labels
    y_pred = clf.predict(data.X, lengths=data.lengths)
    assert np.isin(y_pred, (0, 1)).all()

    # Predict probabilities
    y_probs = clf.predict_proba(data.X, lengths=data.lengths)
    assert ((y_probs >= 0) & (y_probs <= 1)).all()
    npt.assert_almost_equal(y_probs.sum(axis=1), 1.0)

    # Predict log probabilities
    clf.predict_log_proba(data.X, lengths=data.lengths)

    # Calculate accuracy
    acc = clf.score(data.X, data.y, lengths=data.lengths)
    assert acc > 0.8
