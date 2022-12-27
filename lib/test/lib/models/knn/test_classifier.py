import os
import math
import pytest
from copy import deepcopy
from tempfile import TemporaryDirectory
from unittest.mock import Mock

import numpy as np

from sequentia.datasets import load_digits
from sequentia.models.knn import KNNClassifier
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
def test_classifier_e2e(request, k, weighting, dataset, random_state):
    clf = KNNClassifier(k=k, weighting=weighting, random_state=random_state)

    assert clf.k == k
    assert clf.weighting == weighting

    data = dataset.copy()
    data._X = data._X[:, :1] # only use one feature
    subset, _ = data.split(test_size=0.98, random_state=random_state, stratify=True)
    train, test = subset.split(test_size=0.2, random_state=random_state, stratify=True)

    assert_fit(clf.fit(*train.X_y_lengths), train)
    params = clf.get_params()

    scores_pred = clf.predict_scores(*test.X_lengths)
    assert scores_pred.shape == (len(test), n_classes)

    proba_pred = clf.predict_proba(*test.X_lengths)
    assert proba_pred.shape == (len(test), n_classes)
    assert_equal(proba_pred.sum(axis=1), 1)
    assert ((proba_pred >= 0) & (proba_pred <= 1)).all()

    y_pred = clf.predict(*test.X_lengths)
    assert np.issubdtype(y_pred.dtype, np.integer)
    assert y_pred.shape == (len(test),)
    assert set(y_pred).issubset(set(range(n_classes)))

    acc = clf.score(*test.X_y_lengths)
    assert 0 <= acc <= 1

    # check serialization/deserialization
    with TemporaryDirectory() as temp_dir:
        model_path = f"{temp_dir}/{request.node.originalname}.model"
        # check that save works
        clf.save(model_path)
        assert os.path.isfile(model_path)
        # check that load works
        clf = KNNClassifier.load(model_path)
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


def test_classifier_predict_train(dataset, random_state):
    """Should be able to perfectly predict training data with k=1"""
    clf = KNNClassifier(k=1, random_state=random_state)

    data = dataset.copy()
    data._X = data._X[:, :1] # only use one feature
    train, _ = data.split(train_size=0.05, random_state=random_state, stratify=True)

    assert_fit(clf.fit(*train.X_y_lengths), train)
    assert math.isclose(clf.score(*train.X_y_lengths), 1.)


@pytest.mark.parametrize('classes', [[0, 1, 2], [2, 0, 1]])
def test_classifier_compute_scores(classes, random_state):
    clf = KNNClassifier(k=5)
    clf.random_state_ = random_state
    clf.classes_ = np.array(classes)

    labels = np.array([
        [0, 2, 1, 2, 2],
        [0, 1, 1, 2, 0],
        [1, 0, 0, 1, 2],
        [0, 0, 0, 1, 1]
    ])
    weightings = np.ones_like(labels)

    scores = clf._compute_scores(labels, weightings)
    if np.allclose(classes, [0, 1, 2]):
        assert_equal(scores, [
            [1, 1, 3],
            [2, 2, 1],
            [2, 2, 1],
            [3, 2, 0]
        ])
    elif np.allclose(classes, [2, 0, 1]):
        assert_equal(scores, [
            [3, 1, 1],
            [1, 2, 2],
            [1, 2, 2],
            [0, 3, 2]
        ])


@pytest.mark.parametrize('classes', [[0, 1, 2], [2, 0, 1]])
def test_classifier_find_max_labels_chunk(classes, random_state):
    clf = KNNClassifier()
    clf.random_state_ = random_state
    clf.classes_ = np.array(classes)

    score_chunk = np.array([
        [10, 20, 20],
        [10, 30, 20],
        [10, 10, 10],
        [10, 10, 20]
    ])

    max_labels = clf._find_max_labels_chunk(score_chunk)
    if np.allclose(classes, [0, 1, 2]):
        assert max_labels[0] in (1, 2)
        assert max_labels[1] == 1
        assert max_labels[2] in (0, 1, 2)
        assert max_labels[3] == 2
    elif np.allclose(classes, [2, 0, 1]):
        assert max_labels[0] in (0, 1)
        assert max_labels[1] == 0
        assert max_labels[2] in (0, 1, 2)
        assert max_labels[3] == 1


@pytest.mark.parametrize('tie', [True, False])
def test_classifier_multi_argmax(tie):
    if tie:
        arr = np.array([3, 2, 4, 1, 3, 4, 4, 0, 2, 4])
        assert_equal(
            KNNClassifier._multi_argmax(arr),
            np.array([2, 5, 6, 9])
        )
    else:
        arr = np.array([3, 2, 1, 1, 3, 4, 1, 0, 2, 0])
        assert_equal(
            KNNClassifier._multi_argmax(arr),
            np.array([5])
        )


@pytest.mark.parametrize('k', [1, 2, 5])
@pytest.mark.parametrize('sort', [True, False])
def test_classifier_query_neighbors(k, sort, dataset, random_state):
    clf = KNNClassifier(k=k, random_state=random_state)

    data = dataset.copy()
    data._X = data._X[:, :1] # only use one feature
    subset, _ = data.split(test_size=0.98, random_state=random_state, stratify=True)
    train, test = subset.split(test_size=0.2, random_state=random_state, stratify=True)

    assert_fit(clf.fit(*train.X_y_lengths), train)

    k_idxs, k_distances, k_labels = clf.query_neighbors(*test.X_lengths, sort=sort)

    # check that indices are between 0 and len(train)
    assert np.issubdtype(k_idxs.dtype, np.integer)
    assert k_idxs.shape == (len(test), clf.k)
    assert set(k_idxs.flatten()).issubset(set(np.arange(len(train))))

    # check that distances are sorted if sort=True
    np.issubdtype(k_distances.dtype, np.floating)
    assert k_distances.shape == (len(test), clf.k)
    if sort and k > 1:
        assert (k_distances[:, 1:] >= k_distances[:, :-1]).all()

    # check that labels are a subset of training labels + check that labels match indices
    assert np.issubdtype(k_labels.dtype, np.integer)
    assert k_labels.shape == (len(test), clf.k)
    assert set(k_labels.flatten()).issubset(set(train.y))
    assert_equal(train.y[k_idxs], k_labels)


def test_classifier_compute_distance_matrix(dataset, random_state):
    clf = KNNClassifier()

    data = dataset.copy()
    data._X = data._X[:, :1] # only use one feature
    subset, _ = data.split(test_size=0.98, random_state=random_state, stratify=True)
    train, test = subset.split(test_size=0.2, random_state=random_state, stratify=True)

    assert_fit(clf.fit(*train.X_y_lengths), train)

    distances = clf.compute_distance_matrix(*test.X_lengths)
    assert distances.shape == (len(test), len(train))


def test_classifier_distance_matrix_row_col_chunk():
    clf = KNNClassifier()

    clf.X_ = np.expand_dims(np.arange(7), axis=-1)
    col_idxs = np.array([[0, 1], [1, 3], [4, 7]]) # lengths = 1, 2, 3

    X = np.expand_dims(np.arange(14), axis=-1)
    row_idxs = np.array([[0, 2], [2, 5], [5, 9], [9, 14]]) # lengths = 2, 3, 4, 5

    distances = clf._distance_matrix_row_col_chunk(col_idxs, row_idxs, X, lambda x1, x2: len(x1) - len(x2))
    assert_equal(
        distances,
        np.array([
            [1, 0, -1],
            [2, 1,  0],
            [3, 2,  1],
            [4, 3,  2]
        ])
    )
