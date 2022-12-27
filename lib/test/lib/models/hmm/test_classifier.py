import os
import pytest
from copy import deepcopy
from tempfile import TemporaryDirectory
from unittest.mock import Mock

import numpy as np

from sequentia.datasets import load_digits, load_gene_families
from sequentia.models.hmm import GaussianMixtureHMM, CategoricalHMM, HMMClassifier
from sequentia.utils.validation import _check_is_fitted

from .variants.test_gaussian_mixture import assert_fit as assert_gaussian_mixture_fit
from .variants.test_categorical import assert_fit as assert_categorical_fit
from ....support.assertions import assert_equal

n_classes = 7


@pytest.fixture(scope='module')
def random_state(request):
    return np.random.RandomState(1)


@pytest.fixture(scope='module')
def dataset(request):
    if request.param == 'digits':
        return load_digits(digits=range(n_classes))
    elif request.param == 'gene_families':
        data, _ = load_gene_families()
        return data


@pytest.fixture(scope='module')
def model(random_state, request):
    if request.param == 'gaussian_mixture':
        return GaussianMixtureHMM(topology='left-right', n_states=2, n_components=1, random_state=random_state)
    elif request.param == 'categorical':
        return CategoricalHMM(topology='left-right', n_states=2, random_state=random_state)


class MockData:
    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length

    @property
    def lengths(self):
        return MockData(self.length)


def assert_fit(clf):
    assert hasattr(clf, 'prior_')
    assert hasattr(clf, 'classes_')
    assert _check_is_fitted(clf, return_=True)

    for hmm in clf.models.values():
        data = MockData(hmm.n_seqs_)
        if isinstance(hmm, GaussianMixtureHMM):
            assert_gaussian_mixture_fit(hmm, data)
        elif isinstance(hmm, CategoricalHMM):
            assert_categorical_fit(hmm, data)


@pytest.mark.parametrize(
    'model, dataset', [
        ('gaussian_mixture', 'digits'),
        ('categorical', 'gene_families')
    ],
    indirect=True
)
@pytest.mark.parametrize(
    'prior', [
        None,
        'frequency',
        {i: (i + 1) / (n_classes * (n_classes + 1) / 2) for i in range(n_classes)}
    ]
)
@pytest.mark.parametrize('prefit', [True, False])
def test_classifier_e2e(request, model, dataset, prior, prefit, random_state):
    clf = HMMClassifier(prior=prior)
    clf.add_models({i: deepcopy(model) for i in range(n_classes)})

    assert clf.prior == prior
    assert len(clf.models) == n_classes
    assert set(clf.models) == set(range(n_classes))
    assert all(isinstance(hmm, type(model)) for hmm in clf.models.values())

    subset, _ = dataset.split(test_size=0.6, random_state=random_state, stratify=True)
    train, test = subset.split(test_size=0.2, random_state=random_state, stratify=True)

    if prefit:
        for X, lengths, c in train.iter_by_class():
            clf.models[c].fit(X, lengths)
        assert_fit(clf.fit())
    else:
        assert_fit(clf.fit(*train.X_y_lengths))

    scores_pred = clf.predict_scores(*test.X_lengths)
    assert scores_pred.shape == (len(test), n_classes)

    proba_pred = clf.predict_proba(*test.X_lengths)
    assert proba_pred.shape == (len(test), n_classes)
    assert_equal(proba_pred.sum(axis=1), 1)
    assert ((proba_pred >= 0) & (proba_pred <= 1)).all()

    y_pred = clf.predict(*test.X_lengths)
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
        clf = HMMClassifier.load(model_path)
        # check that loaded model is fitted
        assert_fit(clf)
        y_pred_load = clf.predict(*test.X_lengths)
        # check that predictions are the same as before serialization
        assert_equal(y_pred, y_pred_load)


@pytest.mark.parametrize('classes', [[0, 1, 2], [2, 0, 1]])
def test_classifier_compute_log_posterior(classes):
    clf = HMMClassifier()
    clf.classes_ = np.array(classes)
    clf.prior_ = {i: np.exp(i) for i in clf.classes_}
    clf.models = {i: Mock(_score=Mock(side_effect=lambda x: 0)) for i in clf.classes_}
    assert_equal(clf._compute_log_posterior(None), clf.classes_)


def test_classifier_compute_scores_chunk():
    clf = HMMClassifier()
    clf.classes_ = np.arange(3)
    clf.prior_ = {i: np.exp(i) for i in clf.classes_}
    clf.models = {i: Mock(_score=Mock(side_effect=len)) for i in clf.classes_}
    X = np.expand_dims(np.arange(10), axis=-1)
    idxs = np.array([[0, 0], [1, 2], [3, 5], [6, 9]]) # lengths = 0, 1, 2, 3
    assert_equal(
        clf._compute_scores_chunk(idxs, X),
        np.tile(np.expand_dims(clf.classes_, axis=-1), len(idxs)).T
        + np.expand_dims(np.arange(len(idxs)), axis=-1)
    )
