import pytest

import hmmlearn
import numpy as np

from sequentia.models import GaussianMixtureHMM
from sequentia.models.hmm.topologies import _topologies
from sequentia.datasets import load_digits
from sequentia.utils.validation import _check_is_fitted

from .....support.assertions import assert_equal, assert_not_equal
from .....support.itertools import combinations


@pytest.fixture(scope='module')
def random_state():
    return np.random.RandomState(0)


@pytest.fixture(scope='module')
def data(random_state):
    data_= load_digits(digits=[0])
    _, subset = data_.split(test_size=0.2, random_state=random_state, stratify=True)
    return subset


@pytest.fixture(scope='module')
def topology(request):
    return _topologies[request.param]


def assert_fit(hmm, data):
    assert hmm.n_seqs_ == len(data.lengths)
    assert (hmm.topology_ is not None) == (hmm.topology is not None)
    assert isinstance(hmm.model, hmmlearn.hmm.GMMHMM)
    assert len(hmm.model.monitor_.history) > 0
    assert _check_is_fitted(hmm, return_=True)


def test_gaussian_mixture_fit_n_states(data, random_state):
    hmm = GaussianMixtureHMM(n_states=7, random_state=random_state)

    assert_fit(hmm.fit(*data.X_lengths), data)

    assert hmm.n_states == 7

    assert hmm.model.startprob_.shape == (hmm.n_states,)
    assert hmm.model.transmat_.shape == (hmm.n_states, hmm.n_states)


def test_gaussian_mixture_fit_n_components(data, random_state):
    hmm = GaussianMixtureHMM(n_components=2, random_state=random_state)

    assert_fit(hmm.fit(*data.X_lengths), data)

    assert hmm.n_components == 2

    assert hmm.model.startprob_.shape == (hmm.n_states,)
    assert hmm.model.transmat_.shape == (hmm.n_states, hmm.n_states)

    n_features = data.X.shape[1]

    assert hmm.model.means_.shape == (hmm.n_states, hmm.n_components, n_features)
    assert hmm.model.covars_.shape == (hmm.n_states, hmm.n_components)
    assert hmm.model.weights_.shape == (hmm.n_states, hmm.n_components)


@pytest.mark.parametrize('covariance_type', ["spherical", "diag", "full", "tied"])
def test_gaussian_mixture_fit_covariance_type(data, random_state, covariance_type):
    hmm = GaussianMixtureHMM(covariance_type=covariance_type, random_state=random_state)

    assert_fit(hmm.fit(*data.X_lengths), data)

    assert hmm.covariance_type == covariance_type

    assert hmm.model.startprob_.shape == (hmm.n_states,)
    assert hmm.model.transmat_.shape == (hmm.n_states, hmm.n_states)

    n_features = data.X.shape[1]

    assert hmm.model.means_.shape == (hmm.n_states, hmm.n_components, n_features)
    assert hmm.model.weights_.shape == (hmm.n_states, hmm.n_components)

    if covariance_type == "spherical":
        assert hmm.model.covars_.shape == (hmm.n_states, hmm.n_components)
    elif covariance_type == "diag":
        assert hmm.model.covars_.shape == (hmm.n_states, hmm.n_components, n_features)
    elif covariance_type == "full":
        assert hmm.model.covars_.shape == (hmm.n_states, hmm.n_components, n_features, n_features)
    elif covariance_type == "tied":
        assert hmm.model.covars_.shape == (hmm.n_states, n_features, n_features)


def test_gaussian_mixture_fit_no_topology(data, random_state):
    hmm = GaussianMixtureHMM(topology=None, random_state=random_state)

    assert_fit(hmm.fit(*data.X_lengths), data)

    assert hmm.topology is None
    assert hmm.topology_ is None

    assert set(hmm.model.init_params) == set('stmcw')
    assert set(hmm.model.params) == set('stmcw')

    assert not hasattr(hmm, '_startprob')
    assert not hasattr(hmm, '_transmat')


@pytest.mark.parametrize('topology', ['ergodic', 'left-right', 'linear'], indirect=True)
@pytest.mark.parametrize('start_probs_type', ['uniform', 'random', None]) # None = custom
def test_gaussian_mixture_fit_set_start_probs(data, random_state, topology, start_probs_type):
    hmm = GaussianMixtureHMM(topology=topology.name, random_state=random_state)
    hmm.set_start_probs(start_probs_type or topology(hmm.n_states, random_state).random_start_probs())

    assert hmm.topology == topology.name
    if start_probs_type is not None:
        assert hmm._startprob == start_probs_type

    assert_fit(hmm.fit(*data.X_lengths), data)

    assert isinstance(hmm.topology_, topology)

    assert set(hmm.model.init_params) == set('mcw')
    assert set(hmm.model.params) == set('stmcw')

    hmm.topology_.check_start_probs(hmm._startprob) # transition matrix before fit
    hmm.topology_.check_start_probs(hmm.model.startprob_) # transition matrix after fit

    if start_probs_type == 'uniform':
        init_startprob = hmm.topology_.uniform_start_probs()
        assert_equal(hmm._startprob, init_startprob) # initial state probabilities should be uniform

    assert_not_equal(hmm._startprob, hmm.model.startprob_) # should update probabilities
    # assert_equal(hmm._startprob == 0, hmm.model.startprob_ == 0) # but locations of zeros (if any) shouldn't change


@pytest.mark.parametrize('topology', ['ergodic', 'left-right', 'linear'], indirect=True)
@pytest.mark.parametrize('transition_type', ['uniform', 'random', None]) # None = custom
def test_gaussian_mixture_fit_set_transitions(data, random_state, topology, transition_type):
    hmm = GaussianMixtureHMM(topology=topology.name, random_state=random_state)
    hmm.set_transitions(transition_type or topology(hmm.n_states, random_state).random_transitions())

    assert hmm.topology == topology.name
    if transition_type is not None:
        assert hmm._transmat == transition_type

    assert_fit(hmm.fit(*data.X_lengths), data)

    assert isinstance(hmm.topology_, topology)

    assert set(hmm.model.init_params) == set('mcw')
    assert set(hmm.model.params) == set('stmcw')

    hmm.topology_.check_transitions(hmm._transmat) # transition matrix before fit
    hmm.topology_.check_transitions(hmm.model.transmat_) # transition matrix after fit

    if transition_type == 'uniform':
        init_transmat = hmm.topology_.uniform_transitions()
        assert_equal(hmm._transmat, init_transmat) # transition probabilities should be uniform

    assert_not_equal(hmm._transmat, hmm.model.transmat_) # should update probabilities
    # assert_equal(hmm._transmat == 0, hmm.model.transmat_ == 0) # but locations of zeros (if any) shouldn't change


@pytest.mark.parametrize('freeze_params', combinations('stmcw'))
def test_gaussian_mixture_fit_freeze_unfreeze(data, random_state, freeze_params):
    hmm = GaussianMixtureHMM(topology='linear', n_components=2, n_states=2, random_state=random_state)
    hmm.freeze(freeze_params)

    assert_fit(hmm.fit(*data.X_lengths), data)

    assert set(hmm.model.params) == set('stmcw') - set(freeze_params)

    hmm.topology_.check_start_probs(hmm._startprob) # initial state dist. before fit
    hmm.topology_.check_start_probs(hmm.model.startprob_) # initial state dist. after fit
    assertion = assert_equal if 's' in freeze_params else assert_not_equal
    assertion(hmm._startprob, hmm.model.startprob_)

    hmm.topology_.check_transitions(hmm._transmat) # transition matrix before fit
    hmm.topology_.check_transitions(hmm.model.transmat_) # transition matrix after fit
    assertion = assert_equal if 't' in freeze_params else assert_not_equal
    assertion(hmm._transmat, hmm.model.transmat_)

    hmm.unfreeze(freeze_params)

    assert_fit(hmm.fit(*data.X_lengths), data)

    assert set(hmm.model.params) == set('stmcw')

    assert_not_equal(hmm._startprob, hmm.model.startprob_)
    assert_not_equal(hmm._transmat, hmm.model.transmat_)
