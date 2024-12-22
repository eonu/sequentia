# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

from __future__ import annotations

import typing as t

import hmmlearn
import numpy as np
import pytest
from _pytest.fixtures import SubRequest

from sequentia import enums
from sequentia._internal import _validation
from sequentia._internal._hmm.topologies import TOPOLOGY_MAP, BaseTopology
from sequentia.datasets import SequentialDataset, load_gene_families
from sequentia.models import CategoricalHMM

from .....conftest import Helpers


@pytest.fixture(scope="module")
def random_state() -> np.random.RandomState:
    return np.random.RandomState(0)


@pytest.fixture(scope="module")
def data(random_state: np.random.RandomState) -> SequentialDataset:
    data_, _ = load_gene_families(families=[0])
    _, subset = data_.split(
        test_size=0.2, random_state=random_state, stratify=True
    )
    return subset


@pytest.fixture(scope="module")
def topology(request: SubRequest) -> BaseTopology:
    return TOPOLOGY_MAP[request.param]


def assert_fit(hmm: CategoricalHMM, /, *, data: SequentialDataset) -> None:
    assert hmm.n_seqs_ == len(data.lengths)
    assert (hmm.topology_ is not None) == (hmm.topology is not None)
    assert isinstance(hmm.model, hmmlearn.hmm.CategoricalHMM)
    assert len(hmm.model.monitor_.history) > 0
    assert _validation.check_is_fitted(hmm, return_=True)


def test_categorical_fit_n_states(
    data: SequentialDataset, random_state: np.random.RandomState
) -> None:
    hmm = CategoricalHMM(n_states=7, random_state=random_state)

    assert_fit(hmm.fit(**data.X_lengths), data=data)

    assert hmm.n_states == 7

    assert hmm.model.startprob_.shape == (hmm.n_states,)
    assert hmm.model.transmat_.shape == (hmm.n_states, hmm.n_states)


def test_categorical_fit_no_topology(
    data: SequentialDataset, random_state: np.random.RandomState
) -> None:
    hmm = CategoricalHMM(topology=None, random_state=random_state)

    assert_fit(hmm.fit(**data.X_lengths), data=data)

    assert hmm.topology is None
    assert hmm.topology_ is None

    assert set(hmm.model.init_params) == set("ste")
    assert set(hmm.model.params) == set("ste")

    assert not hasattr(hmm, "_startprob")
    assert not hasattr(hmm, "_transmat")


@pytest.mark.parametrize("topology", list(enums.TopologyMode), indirect=True)
@pytest.mark.parametrize(
    "start_probs_mode", [*list(enums.TransitionMode), None]
)
def test_categorical_fit_set_state_start_probs(
    helpers: t.Any,
    data: SequentialDataset,
    random_state: np.random.RandomState,
    topology: BaseTopology,
    start_probs_mode: enums.TransitionMode | None,
) -> None:
    hmm = CategoricalHMM(topology=topology.mode, random_state=random_state)
    hmm.set_state_start_probs(
        start_probs_mode
        or topology(
            n_states=hmm.n_states, random_state=random_state
        ).random_start_probs()
    )

    assert hmm.topology == topology.mode
    if start_probs_mode is not None:
        assert hmm._startprob == start_probs_mode

    assert_fit(hmm.fit(**data.X_lengths), data=data)

    assert isinstance(hmm.topology_, topology)

    assert set(hmm.model.init_params) == set("e")
    assert set(hmm.model.params) == set("ste")

    hmm.topology_.check_start_probs(
        hmm._startprob
    )  # transition matrix before fit
    hmm.topology_.check_start_probs(
        hmm.model.startprob_
    )  # transition matrix after fit

    if start_probs_mode == enums.TransitionMode.UNIFORM:
        init_startprob = hmm.topology_.uniform_start_probs()
        helpers.assert_equal(
            hmm._startprob, init_startprob
        )  # initial state probabilities should be uniform

    helpers.assert_not_equal(
        hmm._startprob, hmm.model.startprob_
    )  # should update probabilities
    helpers.assert_equal(
        hmm._startprob == 0, hmm.model.startprob_ == 0
    )  # but locations of zeros (if any) shouldn't change


@pytest.mark.parametrize("topology", list(enums.TopologyMode), indirect=True)
@pytest.mark.parametrize(
    "transition_mode", [*list(enums.TransitionMode), None]
)  # None = custom
def test_categorical_fit_set_state_transition_probs(
    helpers: t.Any,
    data: SequentialDataset,
    random_state: np.random.RandomState,
    topology: BaseTopology,
    transition_mode: enums.TransitionMode | None,
) -> None:
    hmm = CategoricalHMM(topology=topology.mode, random_state=random_state)
    hmm.set_state_transition_probs(
        transition_mode
        or topology(
            n_states=hmm.n_states, random_state=random_state
        ).random_transition_probs()
    )

    assert hmm.topology == topology.mode
    if transition_mode is not None:
        assert hmm._transmat == transition_mode

    assert_fit(hmm.fit(**data.X_lengths), data=data)

    assert isinstance(hmm.topology_, topology)

    assert set(hmm.model.init_params) == set("e")
    assert set(hmm.model.params) == set("ste")

    hmm.topology_.check_transition_probs(
        hmm._transmat
    )  # transition matrix before fit
    hmm.topology_.check_transition_probs(
        hmm.model.transmat_
    )  # transition matrix after fit

    if transition_mode == enums.TransitionMode.UNIFORM:
        init_transmat = hmm.topology_.uniform_transition_probs()
        helpers.assert_equal(
            hmm._transmat, init_transmat
        )  # transition probabilities should be uniform

    helpers.assert_not_equal(
        hmm._transmat, hmm.model.transmat_
    )  # should update probabilities
    helpers.assert_equal(
        hmm._transmat == 0, hmm.model.transmat_ == 0
    )  # but locations of zeros (if any) shouldn't change


@pytest.mark.parametrize("freeze_params", Helpers.combinations("ste"))
def test_categorical_fit_freeze_unfreeze(
    helpers: t.Any,
    data: SequentialDataset,
    random_state: np.random.RandomState,
    freeze_params: str,
) -> None:
    hmm = CategoricalHMM(
        topology=enums.TopologyMode.LINEAR,
        n_states=2,
        random_state=random_state,
    )
    hmm.freeze(freeze_params)

    assert_fit(hmm.fit(**data.X_lengths), data=data)

    assert set(hmm.model.params) == set("ste") - set(freeze_params)

    hmm.topology_.check_start_probs(
        hmm._startprob
    )  # initial state dist. before fit
    hmm.topology_.check_start_probs(
        hmm.model.startprob_
    )  # initial state dist. after fit
    assertion = (
        helpers.assert_equal
        if "s" in freeze_params
        else helpers.assert_not_equal
    )
    assertion(hmm._startprob, hmm.model.startprob_)

    hmm.topology_.check_transition_probs(
        hmm._transmat
    )  # transition matrix before fit
    hmm.topology_.check_transition_probs(
        hmm.model.transmat_
    )  # transition matrix after fit
    assertion = (
        helpers.assert_equal
        if "t" in freeze_params
        else helpers.assert_not_equal
    )
    assertion(hmm._transmat, hmm.model.transmat_)

    hmm.unfreeze(freeze_params)

    assert_fit(hmm.fit(**data.X_lengths), data=data)

    assert set(hmm.model.params) == set("ste")

    helpers.assert_not_equal(hmm._startprob, hmm.model.startprob_)
    helpers.assert_not_equal(hmm._transmat, hmm.model.transmat_)
