import os
import math
import pytest
from copy import deepcopy
from tempfile import TemporaryDirectory
from unittest.mock import Mock

import numpy as np

from sequentia.datasets import load_digits
from sequentia.models.knn import KNNRegressor
from sequentia.utils.validation import _check_is_fitted

from ....support.assertions import assert_equal

n_classes = 3


@pytest.fixture(scope='module')
def random_state(request):
    return np.random.RandomState(1)


@pytest.fixture(scope='module')
def dataset():
    return load_digits(digits=range(n_classes))


def assert_fit(clf, data):
    assert hasattr(clf, 'X_')
    assert hasattr(clf, 'y_')
    assert hasattr(clf, 'lengths_')
    assert hasattr(clf, 'idxs_')
    assert _check_is_fitted(clf, return_=True)
    assert_equal(clf.X_, data.X)
    assert_equal(clf.y_, data.y)
    assert_equal(clf.lengths_, data.lengths)


@pytest.mark.parametrize('k', [1, 2, 5])
@pytest.mark.parametrize('weighting', [None, lambda x: np.exp(-x)])
def test_regressor_e2e(request, k, weighting, dataset, random_state):
    clf = KNNRegressor(k=k, weighting=weighting, random_state=random_state)

    assert clf.k == k
    assert clf.weighting == weighting

    data = dataset.copy()
    data._X = data._X[:, :1] # only use one feature
    subset, _ = data.split(test_size=0.98, random_state=random_state, stratify=True)
    train, test = subset.split(test_size=0.2, random_state=random_state, stratify=True)

    assert_fit(clf.fit(*train.X_y_lengths), train)
    params = clf.get_params()

    y_pred = clf.predict(*test.X_lengths)
    assert np.issubdtype(y_pred.dtype, np.floating)
    assert y_pred.shape == (len(test),)

    acc = clf.score(*test.X_y_lengths)

    # check serialization/deserialization
    with TemporaryDirectory() as temp_dir:
        model_path = f"{temp_dir}/{request.node.originalname}.model"
        # check that save works
        clf.save(model_path)
        assert os.path.isfile(model_path)
        # check that load works
        clf = KNNRegressor.load(model_path)
        assert (set(clf.get_params().keys()) - set(['weighting'])) == (set(params.keys()) - set(['weighting']))
        # sanity check that custom weighting functions are the same
        if weighting:
            x = random_state.rand(100)
            assert_equal(weighting(x), clf.weighting(x))
        # check that loaded model is fitted and can make predictions
        assert_fit(clf, train)
        y_pred_load = clf.predict(*test.X_lengths)
        if k == 1:
            # predictions should be same as before
            assert_equal(y_pred, y_pred_load)


def test_regressor_predict_train(dataset, random_state):
    """Should be able to perfectly predict training data with k=1"""
    clf = KNNRegressor(k=1, random_state=random_state)

    data = dataset.copy()
    data._X = data._X[:, :1] # only use one feature
    train, _ = data.split(train_size=0.05, random_state=random_state, stratify=True)

    assert_fit(clf.fit(*train.X_y_lengths), train)
    assert math.isclose(clf.score(*train.X_y_lengths), 1.)


def test_regressor_weighting(random_state):
    clf = KNNRegressor(k=3, weighting=lambda x: np.where(x > 10, 0.5, 1))
    clf.random_state_ = random_state

    clf.query_neighbors = Mock(
        return_value=(
            None,
            np.array([
                [1.5, 2, 1],
                [2.5, 1, 0.5],
            ]),
            np.array([
                [10.2, 11.5, 10.4],
                [8.0, 6.5, 5.5]
            ])
        )
    )

    assert_equal(
        clf.predict(None, None),
        np.array([
            (10.2 * 0.5 + 11.5 * 0.5 + 10.4 * 0.5) / (0.5 * clf.k),
            (8.0 * 1 + 6.5 * 1 + 5.5 * 1) / (1 * clf.k)
        ])
    )


@pytest.mark.parametrize('k', [1, 2, 5])
@pytest.mark.parametrize('sort', [True, False])
def test_regressor_query_neighbors(k, sort, dataset, random_state):
    clf = KNNRegressor(k=k, random_state=random_state)

    data = dataset.copy()
    data._X = data._X[:, :1] # only use one feature

    subset, _ = data.split(test_size=0.98, random_state=random_state, stratify=True)
    train, test = subset.split(test_size=0.2, random_state=random_state, stratify=True)

    clf.fit(*train.X_y_lengths)

    k_idxs, k_distances, k_outputs = clf.query_neighbors(*test.X_lengths, sort=sort)

    # check that indices are between 0 and len(train)
    assert np.issubdtype(k_idxs.dtype, np.integer)
    assert k_idxs.shape == (len(test), clf.k)
    assert set(k_idxs.flatten()).issubset(set(np.arange(len(train))))

    # check that distances are sorted if sort=True
    np.issubdtype(k_distances.dtype, np.floating)
    assert k_distances.shape == (len(test), clf.k)
    if sort and k > 1:
        assert (k_distances[:, 1:] >= k_distances[:, :-1]).all()

    # check that labels are a subset of training outputs + check that outputs match indices
    assert np.issubdtype(k_outputs.dtype, np.floating)
    assert k_outputs.shape == (len(test), clf.k)
    assert set(k_outputs.flatten()).issubset(set(train.y))
    assert_equal(train.y[k_idxs], k_outputs)
