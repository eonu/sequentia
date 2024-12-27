# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

import typing as t

import numpy as np
import pytest

from sequentia._internal._hmm import topologies


@pytest.fixture(scope="module")
def random_state() -> np.random.RandomState:
    return np.random.RandomState(0)


def test_base_uniform_start_probs_min(
    helpers: t.Any, random_state: np.random.RandomState
) -> None:
    """Generate a uniform initial state distribution with the minimum number of states"""
    topology = topologies.BaseTopology(n_states=1, random_state=random_state)
    start_probs = topology.uniform_start_probs()
    helpers.assert_distribution(start_probs)
    helpers.assert_equal(start_probs, np.array([1.0]))


def test_base_uniform_start_probs_small(
    helpers: t.Any, random_state: np.random.RandomState
) -> None:
    """Generate a uniform initial state distribution with a few states"""
    topology = topologies.BaseTopology(n_states=2, random_state=random_state)
    start_probs = topology.uniform_start_probs()
    helpers.assert_distribution(start_probs)
    helpers.assert_equal(start_probs, np.array([0.5, 0.5]))


def test_base_uniform_start_probs_many(
    helpers: t.Any, random_state: np.random.RandomState
) -> None:
    """Generate a uniform initial state distribution with many states"""
    topology = topologies.BaseTopology(n_states=5, random_state=random_state)
    start_probs = topology.uniform_start_probs()
    helpers.assert_distribution(start_probs)
    helpers.assert_equal(start_probs, np.array([0.2, 0.2, 0.2, 0.2, 0.2]))


def test_base_random_start_probs_min(
    helpers: t.Any, random_state: np.random.RandomState
) -> None:
    """Generate a random initial state distribution with minimal states"""
    topology = topologies.BaseTopology(n_states=1, random_state=random_state)
    start_probs = topology.random_start_probs()
    helpers.assert_distribution(start_probs)
    helpers.assert_equal(start_probs, np.array([1.0]))


def test_base_random_start_probs_small(
    helpers: t.Any, random_state: np.random.RandomState
) -> None:
    """Generate a random initial state distribution with few states"""
    topology = topologies.BaseTopology(n_states=2, random_state=random_state)
    start_probs = topology.random_start_probs()
    helpers.assert_distribution(start_probs)
    helpers.assert_equal(start_probs, np.array([0.57633871, 0.42366129]))


def test_base_random_initial_many(
    helpers: t.Any, random_state: np.random.RandomState
) -> None:
    """Generate a random initial state distribution with many states"""
    topology = topologies.BaseTopology(n_states=5, random_state=random_state)
    start_probs = topology.random_start_probs()
    helpers.assert_distribution(start_probs)
    helpers.assert_equal(
        start_probs,
        np.array([0.15210286, 0.10647349, 0.20059295, 0.11120171, 0.42962898]),
    )


def test_left_right_uniform_transitions_min(
    helpers: t.Any, random_state: np.random.RandomState
) -> None:
    """Generate a uniform left-right transition matrix with minimal states"""
    topology = topologies.LeftRightTopology(
        n_states=1, random_state=random_state
    )
    transitions = topology.uniform_transition_probs()
    helpers.assert_distribution(transitions)
    helpers.assert_equal(transitions, np.array([[1.0]]))


def test_left_right_uniform_transitions_small(
    helpers: t.Any, random_state: np.random.RandomState
) -> None:
    """Generate a uniform left-right transition matrix with few states"""
    topology = topologies.LeftRightTopology(
        n_states=2, random_state=random_state
    )
    transitions = topology.uniform_transition_probs()
    helpers.assert_distribution(transitions)
    helpers.assert_equal(
        transitions,
        np.array(
            [
                [0.5, 0.5],
                [0.0, 1.0],
            ]
        ),
    )


def test_left_right_uniform_transitions_many(
    helpers: t.Any, random_state: np.random.RandomState
) -> None:
    """Generate a uniform left-right transition matrix with many states"""
    topology = topologies.LeftRightTopology(
        n_states=5, random_state=random_state
    )
    transitions = topology.uniform_transition_probs()
    helpers.assert_distribution(transitions)
    helpers.assert_equal(
        transitions,
        np.array(
            [
                [0.2, 0.2, 0.2, 0.2, 0.2],
                [0.0, 0.25, 0.25, 0.25, 0.25],
                [0.0, 0.0, 0.33333333, 0.33333333, 0.33333333],
                [0.0, 0.0, 0.0, 0.5, 0.5],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )


def test_left_right_random_transitions_min(
    helpers: t.Any, random_state: np.random.RandomState
) -> None:
    """Generate a random left-right transition matrix with minimal states"""
    topology = topologies.LeftRightTopology(
        n_states=1, random_state=random_state
    )
    transitions = topology.random_transition_probs()
    helpers.assert_distribution(transitions)
    helpers.assert_equal(transitions, np.array([[1.0]]))


def test_left_right_random_transitions_small(
    helpers: t.Any, random_state: np.random.RandomState
) -> None:
    """Generate a random left-right transition matrix with few states"""
    topology = topologies.LeftRightTopology(
        n_states=2, random_state=random_state
    )
    transitions = topology.random_transition_probs()
    helpers.assert_distribution(transitions)
    helpers.assert_equal(
        transitions,
        np.array(
            [
                [0.23561633, 0.76438367],
                [0.0, 1.0],
            ]
        ),
    )


def test_left_right_random_transitions_many(
    helpers: t.Any, random_state: np.random.RandomState
) -> None:
    """Generate a random left-right transition matrix with many states"""
    topology = topologies.LeftRightTopology(
        n_states=5, random_state=random_state
    )
    transitions = topology.random_transition_probs()
    helpers.assert_distribution(transitions)
    helpers.assert_equal(
        transitions,
        np.array(
            [
                [0.23169814, 0.71716356, 0.02033845, 0.02516204, 0.00563782],
                [0.0, 0.19474072, 0.16405008, 0.22228532, 0.41892388],
                [0.0, 0.0, 0.42912755, 0.16545797, 0.40541448],
                [0.0, 0.0, 0.0, 0.109713, 0.890287],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )


def test_left_right_check_transitions_invalid(
    random_state: np.random.RandomState,
) -> None:
    """Validate an invalid left-right transition matrix"""
    topology = topologies.LeftRightTopology(
        n_states=5, random_state=random_state
    )
    transitions = topologies.ErgodicTopology(
        n_states=5, random_state=random_state
    ).random_transition_probs()
    with pytest.raises(ValueError) as e:  # noqa: PT011
        topology.check_transition_probs(transitions)
    assert (
        str(e.value) == "Left-right transition matrix must be upper-triangular"
    )


def test_left_right_check_transitions_valid(
    random_state: np.random.RandomState,
) -> None:
    """Validate a valid left-right transition matrix"""
    topology = topologies.LeftRightTopology(
        n_states=5, random_state=random_state
    )
    transitions = topology.random_transition_probs()
    topology.check_transition_probs(transitions)


def test_ergodic_uniform_transitions_min(
    helpers: t.Any, random_state: np.random.RandomState
) -> None:
    """Generate a uniform ergodic transition matrix with minimal states"""
    topology = topologies.ErgodicTopology(
        n_states=1, random_state=random_state
    )
    transitions = topology.uniform_transition_probs()
    helpers.assert_distribution(transitions)
    helpers.assert_equal(transitions, np.array([[1.0]]))


def test_ergodic_uniform_transitions_small(
    helpers: t.Any, random_state: np.random.RandomState
) -> None:
    """Generate a uniform ergodic transition matrix with few states"""
    topology = topologies.ErgodicTopology(
        n_states=2, random_state=random_state
    )
    transitions = topology.uniform_transition_probs()
    helpers.assert_distribution(transitions)
    helpers.assert_equal(
        transitions,
        np.array(
            [
                [0.5, 0.5],
                [0.5, 0.5],
            ]
        ),
    )


def test_ergodic_uniform_transitions_many(
    helpers: t.Any, random_state: np.random.RandomState
) -> None:
    """Generate a uniform ergodic transition matrix with many states"""
    topology = topologies.ErgodicTopology(
        n_states=5, random_state=random_state
    )
    transitions = topology.uniform_transition_probs()
    helpers.assert_distribution(transitions)
    helpers.assert_equal(
        transitions,
        np.array(
            [
                [0.2, 0.2, 0.2, 0.2, 0.2],
                [0.2, 0.2, 0.2, 0.2, 0.2],
                [0.2, 0.2, 0.2, 0.2, 0.2],
                [0.2, 0.2, 0.2, 0.2, 0.2],
                [0.2, 0.2, 0.2, 0.2, 0.2],
            ]
        ),
    )


def test_ergodic_random_transitions_min(
    helpers: t.Any, random_state: np.random.RandomState
) -> None:
    """Generate a random ergodic transition matrix with minimal states"""
    topology = topologies.ErgodicTopology(
        n_states=1, random_state=random_state
    )
    transitions = topology.random_transition_probs()
    helpers.assert_distribution(transitions)
    helpers.assert_equal(transitions, np.array([[1.0]]))


def test_ergodic_random_transitions_small(
    helpers: t.Any, random_state: np.random.RandomState
) -> None:
    """Generate a random ergodic transition matrix with few states"""
    topology = topologies.ErgodicTopology(
        n_states=2, random_state=random_state
    )
    transitions = topology.random_transition_probs()
    helpers.assert_distribution(transitions)
    helpers.assert_equal(
        transitions,
        np.array(
            [
                [0.9474011, 0.0525989],
                [0.85567599, 0.14432401],
            ]
        ),
    )


def test_ergodic_random_transitions_many(
    helpers: t.Any, random_state: np.random.RandomState
) -> None:
    """Generate a random ergodic transition matrix with many states"""
    topology = topologies.ErgodicTopology(
        n_states=5, random_state=random_state
    )
    transitions = topology.random_transition_probs()
    helpers.assert_distribution(transitions)
    helpers.assert_equal(
        transitions,
        np.array(
            [
                [0.58715548, 0.14491542, 0.20980762, 0.00623944, 0.05188205],
                [0.0840705, 0.23055049, 0.08297536, 0.25124688, 0.35115677],
                [0.02117615, 0.37664662, 0.26705912, 0.09851123, 0.23660688],
                [0.01938041, 0.16853843, 0.52046123, 0.07535256, 0.21626737],
                [0.04996846, 0.44545843, 0.12079423, 0.07154241, 0.31223646],
            ]
        ),
    )


def test_ergodic_check_transitions_invalid(
    random_state: np.random.RandomState
) -> None:
    """Validate an invalid ergodic transition matrix"""
    topology = topologies.ErgodicTopology(
        n_states=5, random_state=random_state
    )
    transitions = topologies.LeftRightTopology(
        n_states=5, random_state=random_state
    ).random_transition_probs()
    with pytest.warns(UserWarning):
        topology.check_transition_probs(transitions)


def test_ergodic_check_transitions_valid(
    random_state: np.random.RandomState
) -> None:
    """Validate a valid ergodic transition matrix"""
    topology = topologies.ErgodicTopology(
        n_states=5, random_state=random_state
    )
    transitions = topology.random_transition_probs()
    topology.check_transition_probs(transitions)


def test_linear_uniform_transitions_min(
    helpers: t.Any, random_state: np.random.RandomState
) -> None:
    """Generate a uniform linear transition matrix with minimal states"""
    topology = topologies.LinearTopology(n_states=1, random_state=random_state)
    transitions = topology.uniform_transition_probs()
    helpers.assert_distribution(transitions)
    helpers.assert_equal(transitions, np.array([[1.0]]))


def test_linear_uniform_transitions_small(
    helpers: t.Any, random_state: np.random.RandomState
) -> None:
    """Generate a uniform linear transition matrix with few states"""
    topology = topologies.LinearTopology(n_states=2, random_state=random_state)
    transitions = topology.uniform_transition_probs()
    helpers.assert_distribution(transitions)
    helpers.assert_equal(
        transitions,
        np.array(
            [
                [0.5, 0.5],
                [0.0, 1.0],
            ]
        ),
    )


def test_linear_uniform_transitions_many(
    helpers: t.Any, random_state: np.random.RandomState
) -> None:
    """Generate a uniform linear transition matrix with many states"""
    topology = topologies.LinearTopology(n_states=5, random_state=random_state)
    transitions = topology.uniform_transition_probs()
    helpers.assert_distribution(transitions)
    helpers.assert_equal(
        transitions,
        np.array(
            [
                [0.5, 0.5, 0.0, 0.0, 0.0],
                [0.0, 0.5, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.5, 0.5, 0.0],
                [0.0, 0.0, 0.0, 0.5, 0.5],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )


def test_linear_random_transitions_min(
    helpers: t.Any, random_state: np.random.RandomState
) -> None:
    """Generate a random linear transition matrix with minimal states"""
    topology = topologies.LinearTopology(n_states=1, random_state=random_state)
    transitions = topology.random_transition_probs()
    helpers.assert_distribution(transitions)
    helpers.assert_equal(transitions, np.array([[1.0]]))


def test_linear_random_transitions_small(
    helpers: t.Any, random_state: np.random.RandomState
) -> None:
    """Generate a random linear transition matrix with few states"""
    topology = topologies.LinearTopology(n_states=2, random_state=random_state)
    transitions = topology.random_transition_probs()
    helpers.assert_distribution(transitions)
    helpers.assert_equal(
        transitions,
        np.array(
            [
                [0.65157396, 0.34842604],
                [0.0, 1.0],
            ]
        ),
    )


def test_linear_random_transitions_many(
    helpers: t.Any, random_state: np.random.RandomState
) -> None:
    """Generate a random linear transition matrix with many states"""
    topology = topologies.LinearTopology(n_states=5, random_state=random_state)
    transitions = topology.random_transition_probs()
    helpers.assert_distribution(transitions)
    helpers.assert_equal(
        transitions,
        np.array(
            [
                [0.44455421, 0.55544579, 0.0, 0.0, 0.0],
                [0.0, 0.57553614, 0.42446386, 0.0, 0.0],
                [0.0, 0.0, 0.92014965, 0.07985035, 0.0],
                [0.0, 0.0, 0.0, 0.66790982, 0.33209018],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )


def test_linear_check_transitions_invalid(
    random_state: np.random.RandomState
) -> None:
    """Validate an invalid linear transition matrix"""
    topology = topologies.LinearTopology(n_states=5, random_state=random_state)
    transitions = topologies.ErgodicTopology(
        n_states=5, random_state=random_state
    ).random_transition_probs()
    with pytest.raises(ValueError) as e:  # noqa: PT011
        topology.check_transition_probs(transitions)
    assert (
        str(e.value) == "Left-right transition matrix must be upper-triangular"
    )


def test_linear_check_transitions_valid(
    random_state: np.random.RandomState
) -> None:
    """Validate a valid linear transition matrix"""
    topology = topologies.LinearTopology(n_states=5, random_state=random_state)
    transitions = topology.random_transition_probs()
    topology.check_transition_probs(transitions)
