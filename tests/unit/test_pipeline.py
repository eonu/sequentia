# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

import typing as t

import numpy as np
import pytest
from _pytest.fixtures import SubRequest
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import scale
from sklearn.utils._param_validation import InvalidParameterError
from sklearn.utils.validation import check_is_fitted

from sequentia.datasets import SequentialDataset, load_digits
from sequentia.models import KNNClassifier
from sequentia.preprocessing import IndependentFunctionTransformer


@pytest.fixture(scope="module")
def random_state(request: SubRequest) -> np.random.RandomState:
    return np.random.RandomState(0)


@pytest.fixture(scope="module")
def data(random_state: np.random.RandomState) -> SequentialDataset:
    data_ = load_digits(digits=[0])
    _, subset = data_.split(
        test_size=0.05, random_state=random_state, stratify=True
    )
    return subset


def test_pipeline_with_transforms(
    helpers: t.Any, data: SequentialDataset
) -> None:
    # create pipeline with a stateless and stateful transform
    pipeline = Pipeline(
        [
            (
                "scale",
                IndependentFunctionTransformer(
                    scale, inverse_func=lambda x: x, check_inverse=False
                ),
            ),
            ("pca", PCA(n_components=1)),
        ]
    )

    # check that transforming without fitting doesn't work
    with pytest.raises((NotFittedError, AttributeError)):
        pipeline.transform(**data.X_lengths)

    # check that fitting without y works
    check_is_fitted(pipeline.fit(**data.X_lengths))
    for estimator in pipeline.named_steps.values():
        check_is_fitted(estimator)

    # check that fitting with y works
    check_is_fitted(pipeline.fit(**data.X_y_lengths))
    for estimator in pipeline.named_steps.values():
        check_is_fitted(estimator)

    # check that transforming after fit works
    Xt = pipeline.transform(**data.X_lengths)
    helpers.assert_not_equal(data.X, Xt)
    assert Xt.shape == (len(data.X), 1)

    # check that fit_transform works
    Xt = pipeline.fit_transform(**data.X_lengths)
    helpers.assert_not_equal(data.X, Xt)
    assert Xt.shape == (len(data.X), 1)

    # check that inverse_transform works
    Xi = pipeline.inverse_transform(Xt, lengths=data.lengths)
    helpers.assert_not_equal(Xt, Xi)

    # check that prediction functions relying on X and lengths don't work
    for func in ("predict", "predict_proba"):
        with pytest.raises(AttributeError):
            getattr(pipeline, func)(**data.X_lengths)

    # check that fit_predict doesn't work
    with pytest.raises(AttributeError):
        pipeline.fit_predict(**data.X_y_lengths)

    # TODO @eonu: currently broken for some reason
    # check that score works if the final transform implements it, with y
    # pipeline.score(data.X, data.y, lengths=data.lengths)
    # pipeline.score(**data.X_y_lengths)

    # TODO @eonu: currently broken for some reason
    # check that score works if the final transform implements it, without y
    # pipeline.score(**data.X_lengths)


def test_pipeline_with_estimator(data: SequentialDataset) -> None:
    pipeline = Pipeline(
        [
            ("knn", KNNClassifier(k=1)),
        ]
    )

    # check that transforming doesn't work
    with pytest.raises(AttributeError):
        pipeline.transform(**data.X_lengths)

    # check that fitting without y doesn't work
    with pytest.raises(InvalidParameterError):
        pipeline.fit(**data.X_lengths)

    # check that fitting with y works
    check_is_fitted(pipeline.fit(**data.X_y_lengths))
    for estimator in pipeline.named_steps.values():
        check_is_fitted(estimator)

    # check that transforming doesn't work
    with pytest.raises(AttributeError):
        pipeline.transform(**data.X_lengths)

    # check that fit_transform doesn't work
    with pytest.raises(AttributeError):
        pipeline.fit_transform(**data.X_lengths)

    # check that inverse_transform doesn't work
    with pytest.raises(AttributeError):
        pipeline.inverse_transform(**data.X_lengths)

    # check that predict works
    y_pred = pipeline.predict(**data.X_lengths)
    assert y_pred.shape == data.y.shape
    assert set(y_pred) == set(data.classes)

    # check that predict_log_proba works
    log_proba_pred = pipeline.predict_log_proba(**data.X_lengths)
    assert log_proba_pred.shape == (len(data), len(data.classes))

    # check that predict_proba works
    proba_pred = pipeline.predict_proba(**data.X_lengths)
    assert proba_pred.shape == (len(data), len(data.classes))

    # check that fit_predict works
    y_pred = pipeline.fit_predict(**data.X_y_lengths)
    # check that all steps are fitted
    check_is_fitted(pipeline.fit(**data.X_y_lengths))
    for estimator in pipeline.named_steps.values():
        check_is_fitted(estimator)
    # check that predictions are valid
    assert y_pred.shape == data.y.shape
    assert set(y_pred) == set(data.classes)

    # check that score with y works
    pipeline.score(**data.X_y_lengths)

    # check that score without y doesn't work
    with pytest.raises(InvalidParameterError):
        pipeline.score(**data.X_lengths)


def test_pipeline_with_transforms_and_estimator(
    helpers: t.Any, data: SequentialDataset
) -> None:
    pipeline = Pipeline(
        [
            (
                "scale",
                IndependentFunctionTransformer(
                    scale, inverse_func=lambda x: x, check_inverse=False
                ),
            ),
            ("pca", PCA(n_components=1)),
            ("knn", KNNClassifier(k=1)),
        ]
    )

    # check that transforming doesn't work
    with pytest.raises(AttributeError):
        pipeline.transform(**data.X_lengths)

    # check that fitting without y doesn't work
    with pytest.raises(InvalidParameterError):
        pipeline.fit(**data.X_lengths)

    # check that fitting with y works
    check_is_fitted(pipeline.fit(**data.X_y_lengths))
    for estimator in pipeline.named_steps.values():
        check_is_fitted(estimator)
    # check that X values were transformed
    helpers.assert_not_equal(data.X, pipeline[-1].X_)
    assert pipeline[-1].X_.shape == (len(data.X), 1)

    # check that transforming doesn't work
    with pytest.raises(AttributeError):
        pipeline.transform(**data.X_lengths)

    # check that fit_transform doesn't work
    with pytest.raises(AttributeError):
        pipeline.fit_transform(**data.X_lengths)

    # check that inverse_transform doesn't work
    with pytest.raises(AttributeError):
        pipeline.inverse_transform(**data.X_lengths)

    # check that predict works
    y_pred = pipeline.predict(**data.X_lengths)
    assert y_pred.shape == data.y.shape
    assert set(y_pred) == set(data.classes)

    # check that predict_log_proba works
    log_proba_pred = pipeline.predict_log_proba(**data.X_lengths)
    assert log_proba_pred.shape == (len(data), len(data.classes))

    # check that predict_proba works
    proba_pred = pipeline.predict_proba(**data.X_lengths)
    assert proba_pred.shape == (len(data), len(data.classes))

    # check that fit_predict works
    y_pred = pipeline.fit_predict(**data.X_y_lengths)
    # check that all steps are fitted
    check_is_fitted(pipeline.fit(**data.X_y_lengths))
    for estimator in pipeline.named_steps.values():
        check_is_fitted(estimator)
    # check that predictions are valid
    assert y_pred.shape == data.y.shape
    assert set(y_pred) == set(data.classes)
    # check that X values were transformed
    helpers.assert_not_equal(data.X, pipeline[-1].X_)
    assert pipeline[-1].X_.shape == (len(data.X), 1)

    # check that score with y works
    pipeline.score(**data.X_y_lengths)

    # check that score without y doesn't work
    with pytest.raises(InvalidParameterError):
        pipeline.score(**data.X_lengths)
