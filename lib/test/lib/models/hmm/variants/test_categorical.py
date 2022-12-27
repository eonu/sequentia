import pytest

import hmmlearn
import numpy as np

from sequentia.models import CategoricalHMM
from sequentia.models.hmm.topologies import _topologies
from sequentia.datasets import load_gene_families

from .....support.assertions import assert_equal, assert_not_equal
from .....support.itertools import combinations


@pytest.fixture(scope='module')
def random_state():
    return np.random.RandomState(0)


@pytest.fixture(scope='module')
def data(random_state):
    data_, _ = load_gene_families(families=[0])
    _, subset = data_.split(test_size=0.2, random_state=random_state, stratify=True)
    return subset


@pytest.fixture(scope='module')
def topology(request):
    return _topologies[request.param]


def assert_fit(hmm, data):
    assert hmm.n_seqs_ == len(data.lengths)
    assert (hmm.topology_ is not None) == (hmm.topology is not None)
    assert isinstance(hmm.model, hmmlearn.hmm.CategoricalHMM)
    assert len(hmm.model.monitor_.history) > 0


def test_categorical_fit_n_states(data, random_state):
    hmm = CategoricalHMM(n_states=7, random_state=random_state)

    assert_fit(hmm.fit(*data.X_lengths), data)

    assert hmm.n_states == 7

    assert hmm.model.startprob_.shape == (hmm.n_states,)
    assert hmm.model.transmat_.shape == (hmm.n_states, hmm.n_states)


def test_categorical_fit_no_topology(data, random_state):
    hmm = CategoricalHMM(topology=None, random_state=random_state)

    assert_fit(hmm.fit(*data.X_lengths), data)

    assert hmm.topology is None
    assert hmm.topology_ is None

    assert set(hmm.model.init_params) == set('ste')
    assert set(hmm.model.params) == set('ste')

    assert not hasattr(hmm, '_startprob')
    assert not hasattr(hmm, '_transmat')


@pytest.mark.parametrize('topology', ['ergodic', 'left-right', 'linear'], indirect=True)
@pytest.mark.parametrize('start_probs_type', ['uniform', 'random', None]) # None = custom
def test_categorical_fit_set_start_probs(data, random_state, topology, start_probs_type):
    hmm = CategoricalHMM(topology=topology.name, random_state=random_state)
    hmm.set_start_probs(start_probs_type or topology(hmm.n_states, random_state).random_start_probs())

    assert hmm.topology == topology.name
    if start_probs_type is not None:
        assert hmm._startprob == start_probs_type

    assert_fit(hmm.fit(*data.X_lengths), data)

    assert isinstance(hmm.topology_, topology)

    assert set(hmm.model.init_params) == set('e')
    assert set(hmm.model.params) == set('ste')

    hmm.topology_.check_start_probs(hmm._startprob) # transition matrix before fit
    hmm.topology_.check_start_probs(hmm.model.startprob_) # transition matrix after fit

    if start_probs_type == 'uniform':
        init_startprob = hmm.topology_.uniform_start_probs()
        assert_equal(hmm._startprob, init_startprob) # initial state probabilities should be uniform

    assert_not_equal(hmm._startprob, hmm.model.startprob_) # should update probabilities
    assert_equal(hmm._startprob == 0, hmm.model.startprob_ == 0) # but locations of zeros (if any) shouldn't change


@pytest.mark.parametrize('topology', ['ergodic', 'left-right', 'linear'], indirect=True)
@pytest.mark.parametrize('transition_type', ['uniform', 'random', None]) # None = custom
def test_categorical_fit_set_transitions(data, random_state, topology, transition_type):
    hmm = CategoricalHMM(topology=topology.name, random_state=random_state)
    hmm.set_transitions(transition_type or topology(hmm.n_states, random_state).random_transitions())

    assert hmm.topology == topology.name
    if transition_type is not None:
        assert hmm._transmat == transition_type

    assert_fit(hmm.fit(*data.X_lengths), data)

    assert isinstance(hmm.topology_, topology)

    assert set(hmm.model.init_params) == set('e')
    assert set(hmm.model.params) == set('ste')

    hmm.topology_.check_transitions(hmm._transmat) # transition matrix before fit
    hmm.topology_.check_transitions(hmm.model.transmat_) # transition matrix after fit

    if transition_type == 'uniform':
        init_transmat = hmm.topology_.uniform_transitions()
        assert_equal(hmm._transmat, init_transmat) # transition probabilities should be uniform

    assert_not_equal(hmm._transmat, hmm.model.transmat_) # should update probabilities
    assert_equal(hmm._transmat == 0, hmm.model.transmat_ == 0) # but locations of zeros (if any) shouldn't change


@pytest.mark.parametrize('freeze_params', combinations('ste'))
def test_categorical_fit_freeze_unfreeze(data, random_state, freeze_params):
    hmm = CategoricalHMM(topology='linear', n_states=2, random_state=random_state)
    hmm.freeze(freeze_params)

    assert_fit(hmm.fit(*data.X_lengths), data)

    assert set(hmm.model.params) == set('ste') - set(freeze_params)

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

    assert set(hmm.model.params) == set('ste')

    assert_not_equal(hmm._startprob, hmm.model.startprob_)
    assert_not_equal(hmm._transmat, hmm.model.transmat_)
