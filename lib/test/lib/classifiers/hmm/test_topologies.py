import pytest, warnings, numpy as np
from sequentia.classifiers import _Topology, _LeftRightTopology, _ErgodicTopology, _LinearTopology
from ....support import assert_equal, assert_all_equal, assert_distribution

# Set seed for reproducible randomness
seed = 0
np.random.seed(seed)
rng = np.random.RandomState(seed)

# ========= #
# _Topology #
# ========= #

# --------------------------- #
# _Topology.uniform_initial() #
# --------------------------- #

def test_uniform_initial_min():
    """Generate a uniform initial state distribution with the minimum number of states"""
    topology = _Topology(n_states=1, random_state=rng)
    initial = topology.uniform_initial()
    assert_distribution(initial)
    assert_equal(initial, np.array([
        1.
    ]))

def test_uniform_initial_small():
    """Generate a uniform initial state distribution with a few states"""
    topology = _Topology(n_states=2, random_state=rng)
    initial = topology.uniform_initial()
    assert_distribution(initial)
    assert_equal(initial, np.array([
        0.5, 0.5
    ]))

def test_uniform_initial_many():
    """Generate a uniform initial state distribution with many states"""
    topology = _Topology(n_states=5, random_state=rng)
    initial = topology.uniform_initial()
    assert_distribution(initial)
    assert_equal(initial, np.array([
        0.2, 0.2, 0.2, 0.2, 0.2
    ]))

# -------------------------- #
# _Topology.random_initial() #
# -------------------------- #

def test_random_initial_min():
    """Generate a random initial state distribution with minimal states"""
    topology = _Topology(n_states=1, random_state=rng)
    initial = topology.random_initial()
    assert_distribution(initial)
    assert_equal(initial, np.array([
        1.
    ]))

def test_random_initial_small():
    """Generate a random initial state distribution with few states"""
    topology = _Topology(n_states=2, random_state=rng)
    initial = topology.random_initial()
    assert_distribution(initial)
    assert_equal(initial, np.array([
        0.57633871, 0.42366129
    ]))

def test_random_initial_many():
    """Generate a random initial state distribution with many states"""
    topology = _Topology(n_states=5, random_state=rng)
    initial = topology.random_initial()
    assert_distribution(initial)
    assert_equal(initial, np.array([
        0.15210286, 0.10647349, 0.20059295, 0.11120171, 0.42962898
    ]))

# ================== #
# _LeftRightTopology #
# ================== #

# ---------------------------------------- #
# _LeftRightTopology.uniform_transitions() #
# ---------------------------------------- #

def test_left_right_uniform_transitions_min():
    """Generate a uniform left-right transition matrix with minimal states"""
    topology = _LeftRightTopology(n_states=1, random_state=rng)
    transitions = topology.uniform_transitions()
    assert_distribution(transitions)
    assert_equal(transitions, np.array([
        [1.]
    ]))

def test_left_right_uniform_transitions_small():
    """Generate a uniform left-right transition matrix with few states"""
    topology = _LeftRightTopology(n_states=2, random_state=rng)
    transitions = topology.uniform_transitions()
    assert_distribution(transitions)
    assert_equal(transitions, np.array([
        [0.5, 0.5],
        [0. , 1. ]
    ]))

def test_left_right_uniform_transitions_many():
    """Generate a uniform left-right transition matrix with many states"""
    topology = _LeftRightTopology(n_states=5, random_state=rng)
    transitions = topology.uniform_transitions()
    assert_distribution(transitions)
    assert_equal(transitions, np.array([
        [0.2, 0.2 , 0.2       , 0.2       , 0.2       ],
        [0. , 0.25, 0.25      , 0.25      , 0.25      ],
        [0. , 0.  , 0.33333333, 0.33333333, 0.33333333],
        [0. , 0.  , 0.        , 0.5       , 0.5       ] ,
        [0. , 0.  , 0.        , 0.        , 1.        ]
    ]))

# --------------------------------------- #
# _LeftRightTopology.random_transitions() #
# --------------------------------------- #

def test_left_right_random_transitions_min():
    """Generate a random left-right transition matrix with minimal states"""
    topology = _LeftRightTopology(n_states=1, random_state=rng)
    transitions = topology.random_transitions()
    assert_distribution(transitions)
    assert_equal(transitions, np.array([
        [1.]
    ]))

def test_left_right_random_transitions_small():
    """Generate a random left-right transition matrix with few states"""
    topology = _LeftRightTopology(n_states=2, random_state=rng)
    transitions = topology.random_transitions()
    assert_distribution(transitions)
    assert_equal(transitions, np.array([
        [0.23561633, 0.76438367],
        [0.        , 1.        ]
    ]))

def test_left_right_random_transitions_many():
    """Generate a random left-right transition matrix with many states"""
    topology = _LeftRightTopology(n_states=5, random_state=rng)
    transitions = topology.random_transitions()
    assert_distribution(transitions)
    assert_equal(transitions, np.array([
        [0.56841967, 0.01612013, 0.01994328, 0.0044685 , 0.39104841],
        [0.        , 0.25134034, 0.43904868, 0.20609306, 0.10351793],
        [0.        , 0.        , 0.27462001, 0.12291279, 0.60246721],
        [0.        , 0.        , 0.        , 0.61951739, 0.38048261],
        [0.        , 0.        , 0.        , 0.        , 1.        ]
    ]))

# ----------------------------------------- #
# _LeftRightTopology.validate_transitions() #
# ----------------------------------------- #

def test_left_right_validate_transitions_invalid():
    """Validate an invalid left-right transition matrix"""
    topology = _LeftRightTopology(n_states=5, random_state=rng)
    transitions = _ErgodicTopology(n_states=5, random_state=rng).random_transitions()
    with pytest.raises(ValueError) as e:
        topology.validate_transitions(transitions)
    assert str(e.value) == 'Left-right transition matrix must be upper-triangular'

def test_left_right_validate_transitions_valid():
    """Validate a valid left-right transition matrix"""
    topology = _LeftRightTopology(n_states=5, random_state=rng)
    transitions = topology.random_transitions()
    topology.validate_transitions(transitions)

# -------------------------------------- #
# _ErgodicTopology.uniform_transitions() #
# -------------------------------------- #

def test_ergodic_uniform_transitions_min():
    """Generate a uniform ergodic transition matrix with minimal states"""
    topology = _ErgodicTopology(n_states=1, random_state=rng)
    transitions = topology.uniform_transitions()
    assert_distribution(transitions)
    assert_equal(transitions, np.array([
        [1.]
    ]))

def test_ergodic_uniform_transitions_small():
    """Generate a uniform ergodic transition matrix with few states"""
    topology = _ErgodicTopology(n_states=2, random_state=rng)
    transitions = topology.uniform_transitions()
    assert_distribution(transitions)
    assert_equal(transitions, np.array([
        [0.5, 0.5],
        [0.5, 0.5]
    ]))

def test_ergodic_uniform_transitions_many():
    """Generate a uniform ergodic transition matrix with many states"""
    topology = _ErgodicTopology(n_states=5, random_state=rng)
    transitions = topology.uniform_transitions()
    assert_distribution(transitions)
    assert_equal(transitions, np.array([
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [0.2, 0.2, 0.2, 0.2, 0.2]
    ]))

# ------------------------------------- #
# _ErgodicTopology.random_transitions() #
# ------------------------------------- #

def test_ergodic_random_transitions_min():
    """Generate a random ergodic transition matrix with minimal states"""
    topology = _ErgodicTopology(n_states=1, random_state=rng)
    transitions = topology.random_transitions()
    assert_distribution(transitions)
    assert_equal(transitions, np.array([
        [1.]
    ]))

def test_ergodic_random_transitions_small():
    """Generate a random ergodic transition matrix with few states"""
    topology = _ErgodicTopology(n_states=2, random_state=rng)
    transitions = topology.random_transitions()
    assert_distribution(transitions)
    assert_equal(transitions, np.array([
        [0.87353002, 0.12646998],
        [0.88622334, 0.11377666]
    ]))

def test_ergodic_random_transitions_many():
    """Generate a random ergodic transition matrix with many states"""
    topology = _ErgodicTopology(n_states=5, random_state=rng)
    transitions = topology.random_transitions()
    assert_distribution(transitions)
    assert_equal(transitions, np.array([
        [0.46537016, 0.12619365, 0.07474032, 0.32619324, 0.00750262],
        [0.38836848, 0.00103519, 0.24911885, 0.06922191, 0.29225557],
        [0.5312161 , 0.04639154, 0.13922816, 0.14542372, 0.13774047],
        [0.0361995 , 0.43772711, 0.08498809, 0.26867251, 0.17241279],
        [0.06373359, 0.30347054, 0.09117514, 0.38445582, 0.1571649 ]
    ]))

# --------------------------------------- #
# _ErgodicTopology.validate_transitions() #
# --------------------------------------- #

def test_ergodic_validate_transitions_invalid():
    """Validate an invalid ergodic transition matrix"""
    topology = _ErgodicTopology(n_states=5, random_state=rng)
    transitions = _LeftRightTopology(n_states=5, random_state=rng).random_transitions()
    with pytest.warns(UserWarning):
        topology.validate_transitions(transitions)

def test_ergodic_validate_transitions_valid():
    """Validate a valid ergodic transition matrix"""
    topology = _ErgodicTopology(n_states=5, random_state=rng)
    transitions = topology.random_transitions()
    topology.validate_transitions(transitions)

# ======================== #
# _LinearTopology #
# ======================== #

# ---------------------------------------------- #
# _LinearTopology.uniform_transitions() #
# ---------------------------------------------- #

def test_linear_uniform_transitions_min():
    """Generate a uniform linear transition matrix with minimal states"""
    topology = _LinearTopology(n_states=1, random_state=rng)
    transitions = topology.uniform_transitions()
    assert_distribution(transitions)
    assert_equal(transitions, np.array([
        [1.]
    ]))

def test_linear_uniform_transitions_small():
    """Generate a uniform linear transition matrix with few states"""
    topology = _LinearTopology(n_states=2, random_state=rng)
    transitions = topology.uniform_transitions()
    assert_distribution(transitions)
    assert_equal(transitions, np.array([
        [0.5, 0.5],
        [0. , 1. ]
    ]))

def test_linear_uniform_transitions_many():
    """Generate a uniform linear transition matrix with many states"""
    topology = _LinearTopology(n_states=5, random_state=rng)
    transitions = topology.uniform_transitions()
    assert_distribution(transitions)
    assert_equal(transitions, np.array([
        [0.5, 0.5 , 0.        , 0.        , 0.        ],
        [0. , 0.5 , 0.5       , 0.        , 0.        ],
        [0. , 0.  , 0.5       , 0.5       , 0.        ],
        [0. , 0.  , 0.        , 0.5       , 0.5       ],
        [0. , 0.  , 0.        , 0.        , 1.        ]
    ]))

# --------------------------------------------- #
# _LinearTopology.random_transitions() #
# --------------------------------------------- #

def test_linear_random_transitions_min():
    """Generate a random linear transition matrix with minimal states"""
    topology = _LinearTopology(n_states=1, random_state=rng)
    transitions = topology.random_transitions()
    assert_distribution(transitions)
    assert_equal(transitions, np.array([
        [1.]
    ]))

def test_linear_random_transitions_small():
    """Generate a random linear transition matrix with few states"""
    topology = _LinearTopology(n_states=2, random_state=rng)
    transitions = topology.random_transitions()
    assert_distribution(transitions)
    assert_equal(transitions, np.array([
        [0.87426829, 0.12573171],
        [0.        , 1.        ]
    ]))

def test_linear_random_transitions_many():
    """Generate a random linear transition matrix with many states"""
    topology = _LinearTopology(n_states=5, random_state=rng)
    transitions = topology.random_transitions()
    assert_distribution(transitions)
    assert_equal(transitions, np.array([
        [0.9294571 , 0.0705429 , 0.        , 0.        , 0.        ],
        [0.        , 0.92269318, 0.07730682, 0.        , 0.        ],
        [0.        , 0.        , 0.86161736, 0.13838264, 0.        ],
        [0.        , 0.        , 0.        , 0.13863688, 0.86136312],
        [0.        , 0.        , 0.        , 0.        , 1.        ]
    ]))

# ----------------------------------------------- #
# _LinearTopology.validate_transitions() #
# ----------------------------------------------- #

def test_linear_validate_transitions_invalid():
    """Validate an invalid linear transition matrix"""
    topology = _LinearTopology(n_states=5, random_state=rng)
    transitions = _ErgodicTopology(n_states=5, random_state=rng).random_transitions()
    with pytest.raises(ValueError) as e:
        topology.validate_transitions(transitions)
    assert str(e.value) == 'Left-right transition matrix must be upper-triangular'

def test_linear_validate_transitions_valid():
    """Validate a valid linear transition matrix"""
    topology = _LinearTopology(n_states=5, random_state=rng)
    transitions = topology.random_transitions()
    topology.validate_transitions(transitions)