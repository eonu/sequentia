import pytest, warnings, os, json, numpy as np
from copy import deepcopy
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    import pomegranate as pg
from sequentia.classifiers import HMM, _LeftRightTopology, _ErgodicTopology, _StrictLeftRightTopology
from ....support import assert_equal, assert_not_equal

# Set seed for reproducible randomness
seed = 0
np.random.seed(seed)
rng = np.random.RandomState(seed)

# Create some sample data
X = [rng.random((10 * i, 3)) for i in range(1, 4)]
x = rng.random((15, 3))

# Unparameterized HMMs
hmm_lr = HMM(label='c1', n_states=5, topology='left-right', random_state=rng)
hmm_e = HMM(label='c1', n_states=5, topology='ergodic', random_state=rng)
hmm_slr = HMM(label='c1', n_states=5, topology='strict-left-right', random_state=rng)

# ================================================== #
# HMM.set_uniform_initial() + HMM.initial (property) #
# ================================================== #

def test_left_right_uniform_initial():
    """Uniform initial state distribution for a left-right HMM"""
    hmm = deepcopy(hmm_lr)
    hmm.set_uniform_initial()
    assert_equal(hmm.initial, np.array([
        0.2, 0.2, 0.2, 0.2, 0.2
    ]))

def test_ergodic_uniform_initial():
    """Uniform initial state distribution for an ergodic HMM"""
    hmm = deepcopy(hmm_e)
    hmm.set_uniform_initial()
    assert_equal(hmm.initial, np.array([
        0.2, 0.2, 0.2, 0.2, 0.2
    ]))

def test_strict_left_right_uniform_initial():
    """Uniform initial state distribution for a strict left-right HMM"""
    hmm = deepcopy(hmm_slr)
    hmm.set_uniform_initial()
    assert_equal(hmm.initial, np.array([
        0.2, 0.2, 0.2, 0.2, 0.2
    ]))

# ================================================= #
# HMM.set_random_initial() + HMM.initial (property) #
# ================================================= #

def test_left_right_random_initial():
    """Random initial state distribution for a left-right HMM"""
    hmm = deepcopy(hmm_lr)
    hmm.set_random_initial()
    assert_equal(hmm.initial, np.array([
        0.35029635, 0.13344569, 0.02784745, 0.33782453, 0.15058597
    ]))

def test_ergodic_random_initial():
    """Random initial state distribution for an ergodic HMM"""
    hmm = deepcopy(hmm_e)
    hmm.set_random_initial()
    assert_equal(hmm.initial, np.array([
        0.35029635, 0.13344569, 0.02784745, 0.33782453, 0.15058597
    ]))

def test_strict_left_right_random_initial():
    """Random initial state distribution for a strict left-right HMM"""
    hmm = deepcopy(hmm_slr)
    hmm.set_random_initial()
    assert_equal(hmm.initial, np.array([
        0.35029635, 0.13344569, 0.02784745, 0.33782453, 0.15058597
    ]))

# ========================================================== #
# HMM.set_uniform_transitions() + HMM.transitions (property) #
# ========================================================== #

def test_left_right_uniform_transitions():
    """Uniform transition matrix for a left-right HMM"""
    hmm = deepcopy(hmm_lr)
    hmm.set_uniform_transitions()
    assert_equal(hmm.transitions, np.array([
        [0.2, 0.2 , 0.2       , 0.2       , 0.2       ],
        [0. , 0.25, 0.25      , 0.25      , 0.25      ],
        [0. , 0.  , 0.33333333, 0.33333333, 0.33333333],
        [0. , 0.  , 0.        , 0.5       , 0.5       ],
        [0. , 0.  , 0.        , 0.        , 1.        ]
    ]))

def test_ergodic_uniform_transitions():
    """Uniform transition matrix for an ergodic HMM"""
    hmm = deepcopy(hmm_e)
    hmm.set_uniform_transitions()
    assert_equal(hmm.transitions, np.array([
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [0.2, 0.2, 0.2, 0.2, 0.2]
    ]))

def test_strict_left_right_uniform_transitions():
    """Uniform transition matrix for a strict left-right HMM"""
    hmm = deepcopy(hmm_slr)
    hmm.set_uniform_transitions()
    assert_equal(hmm.transitions, np.array([
        [0.5, 0.5, 0. , 0. , 0. ],
        [0. , 0.5, 0.5, 0. , 0. ],
        [0. , 0. , 0.5, 0.5, 0. ],
        [0. , 0. , 0. , 0.5, 0.5],
        [0. , 0. , 0. , 0. , 1. ]
    ]))

# ========================================================= #
# HMM.set_random_transitions() + HMM.transitions (property) #
# ========================================================= #

def test_left_right_random_transitions():
    """Random transition matrix for a left-right HMM"""
    hmm = deepcopy(hmm_lr)
    hmm.set_random_transitions()
    assert_equal(hmm.transitions, np.array([
        [0.35029635, 0.13344569, 0.02784745, 0.33782453, 0.15058597],
        [0.        , 0.20580715, 0.5280311 , 0.06521685, 0.20094491],
        [0.        , 0.        , 0.33761567, 0.2333124 , 0.42907193],
        [0.        , 0.        , 0.        , 0.36824778, 0.63175222],
        [0.        , 0.        , 0.        , 0.        , 1.        ]
    ]))

def test_ergodic_random_transitions():
    """Random transition matrix for an ergodic HMM"""
    hmm = deepcopy(hmm_e)
    hmm.set_random_transitions()
    assert_equal(hmm.transitions, np.array([
        [0.35029635, 0.13344569, 0.02784745, 0.33782453, 0.15058597],
        [0.19252534, 0.15767581, 0.47989976, 0.01708551, 0.15281357],
        [0.19375092, 0.16425506, 0.21828034, 0.11397708, 0.30973661],
        [0.46906977, 0.02941216, 0.17137502, 0.0333193 , 0.29682374],
        [0.21312406, 0.35221103, 0.08556524, 0.06613143, 0.28296824]
    ]))

def test_strict_left_right_random_transitions():
    """Random transition matrix for a strict left-right HMM"""
    hmm = deepcopy(hmm_slr)
    hmm.set_random_transitions()
    assert_equal(hmm.transitions, np.array([
        [0.72413873, 0.27586127, 0.        , 0.        , 0.        ],
        [0.        , 0.07615418, 0.92384582, 0.        , 0.        ],
        [0.        , 0.        , 0.81752797, 0.18247203, 0.        ],
        [0.        , 0.        , 0.        , 0.24730529, 0.75269471],
        [0.        , 0.        , 0.        , 0.        , 1.        ]
    ]))

# ====================== #
# HMM.fit() + HMM.n_seqs #
# ====================== #

def test_fit_without_initial_and_transition():
    """Fitting before setting the initial state distribution and transition matrix"""
    hmm = deepcopy(hmm_lr)
    with warnings.catch_warnings(), pytest.raises(AttributeError) as e:
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        hmm.fit(X)
    assert str(e.value) == 'Must specify initial state distribution and transitions before the HMM can be fitted'

def test_fit_without_transitions():
    """Fitting before setting the initial transition matrix"""
    hmm = deepcopy(hmm_lr)
    hmm.set_uniform_initial()
    with warnings.catch_warnings(), pytest.raises(AttributeError) as e:
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        hmm.fit(X)
    assert str(e.value) == 'Must specify initial state distribution and transitions before the HMM can be fitted'

def test_fit_without_initial():
    """Fitting before setting the initial state distribution"""
    hmm = deepcopy(hmm_lr)
    hmm.set_uniform_transitions()
    with warnings.catch_warnings(), pytest.raises(AttributeError) as e:
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        hmm.fit(X)
    assert str(e.value) == 'Must specify initial state distribution and transitions before the HMM can be fitted'

def test_fit_sets_internals():
    """Check that fitting sets internal attributes"""
    hmm = deepcopy(hmm_lr)
    hmm.set_uniform_initial()
    hmm.set_uniform_transitions()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        hmm.fit(X)
    assert hmm.n_seqs == 3
    assert isinstance(hmm._model, pg.HiddenMarkovModel)

def test_left_right_fit_updates_initial():
    """Check that fitting updates the initial state distribution of a left-right HMM"""
    hmm = deepcopy(hmm_lr)
    hmm.set_uniform_initial()
    hmm.set_uniform_transitions()
    assert_equal(hmm.initial, np.array([
        0.2, 0.2, 0.2, 0.2, 0.2
    ]))
    before = hmm.initial
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        hmm.fit(X)
    assert_not_equal(hmm.initial, before)

def test_left_right_fit_updates_transitions():
    """Check that fitting updates the transition matrix of a left-right HMM"""
    hmm = deepcopy(hmm_lr)
    hmm.set_uniform_initial()
    hmm.set_uniform_transitions()
    before = hmm.transitions
    assert_equal(hmm.transitions, np.array([
        [0.2       , 0.2       , 0.2       , 0.2       , 0.2       ],
        [0.        , 0.25      , 0.25      , 0.25      , 0.25      ],
        [0.        , 0.        , 0.33333333, 0.33333333, 0.33333333],
        [0.        , 0.        , 0.        , 0.5       , 0.5       ],
        [0.        , 0.        , 0.        , 0.        , 1.        ]
    ]))
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        hmm.fit(X)
    assert_not_equal(hmm.transitions, before)

def test_ergodic_fit_doesnt_updates_initial():
    """Check that fitting does not update the initial state distribution of an ergodic HMM"""
    hmm = deepcopy(hmm_e)
    hmm.set_uniform_initial()
    hmm.set_uniform_transitions()
    assert_equal(hmm.initial, np.array([
        0.2, 0.2, 0.2, 0.2, 0.2
    ]))
    before = hmm.initial
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        hmm.fit(X)
    assert_equal(hmm.initial, before)

def test_ergodic_fit_doesnt_updates_transitions():
    """Check that fitting does not update the transition matrix of an ergodic HMM"""
    hmm = deepcopy(hmm_e)
    hmm.set_uniform_initial()
    hmm.set_uniform_transitions()
    assert_equal(hmm.transitions, np.array([
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [0.2, 0.2, 0.2, 0.2, 0.2]
    ]))
    before = hmm.transitions
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        hmm.fit(X)
    assert_equal(hmm.transitions, before)

def test_strict_left_right_fit_updates_initial():
    """Check that fitting updates the initial state distribution of a strict left-right HMM"""
    hmm = deepcopy(hmm_slr)
    hmm.set_uniform_initial()
    hmm.set_uniform_transitions()
    assert_equal(hmm.initial, np.array([
        0.2, 0.2, 0.2, 0.2, 0.2
    ]))
    before = hmm.initial
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        hmm.fit(X)
    assert_not_equal(hmm.initial, before)

def test_strict_left_right_fit_updates_transitions():
    """Check that fitting updates the transition matrix of a strict left-right HMM"""
    hmm = deepcopy(hmm_slr)
    hmm.set_uniform_initial()
    hmm.set_uniform_transitions()
    before = hmm.transitions
    assert_equal(hmm.transitions, np.array([
        [0.5       , 0.5       , 0.        , 0.        , 0.        ],
        [0.        , 0.5       , 0.5       , 0.        , 0.        ],
        [0.        , 0.        , 0.5       , 0.5       , 0.        ],
        [0.        , 0.        , 0.        , 0.5       , 0.5       ],
        [0.        , 0.        , 0.        , 0.        , 1.        ]
    ]))
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        hmm.fit(X)
    assert_not_equal(hmm.transitions, before)

# ============= #
# HMM.forward() #
# ============= #

def test_forward_without_fit():
    """Forward algorithm without fitting the HMM"""
    hmm = deepcopy(hmm_lr)
    hmm.set_random_initial()
    hmm.set_random_transitions()
    with pytest.raises(AttributeError) as e:
        hmm.forward(x)
    assert str(e.value) == 'The model must be fitted before running the forward algorithm'

def test_left_right_forward():
    """Forward algorithm on a left-right HMM"""
    hmm = deepcopy(hmm_lr)
    hmm.set_random_initial()
    hmm.set_random_transitions()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        hmm.fit(X)
    assert isinstance(hmm.forward(x), float)

def test_ergodic_forward():
    """Forward algorithm on an ergodic HMM"""
    hmm = deepcopy(hmm_e)
    hmm.set_random_initial()
    hmm.set_random_transitions()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        hmm.fit(X)
    assert isinstance(hmm.forward(x), float)

def test_strict_left_right_forward():
    """Forward algorithm on a strict left-right HMM"""
    hmm = deepcopy(hmm_slr)
    hmm.set_random_initial()
    hmm.set_random_transitions()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        hmm.fit(X)
    assert isinstance(hmm.forward(x), float)

# ==================== #
# HMM.label (property) #
# ==================== #

def test_label():
    assert deepcopy(hmm_lr).label == 'c1'

# ======================= #
# HMM.n_states (property) #
# ======================= #

def test_n_states():
    assert deepcopy(hmm_lr).n_states == 5

# ===================== #
# HMM.n_seqs (property) #
# ===================== #

def test_n_seqs_without_fit():
    """Number of sequences without fitting the HMM"""
    hmm = deepcopy(hmm_lr)
    hmm.set_random_initial()
    hmm.set_random_transitions()
    with pytest.raises(AttributeError) as e:
        hmm.n_seqs
    assert str(e.value) == 'The model has not been fitted and has not seen any observation sequences'

def test_n_seqs_with_fit():
    """Number of sequences after fitting the HMM"""
    hmm = deepcopy(hmm_lr)
    hmm.set_random_initial()
    hmm.set_random_transitions()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        hmm.fit(X)
    assert hmm.n_seqs == 3

# ====================== #
# HMM.initial (property) #
# ====================== #

def test_initial_without_setting():
    """Get initial state distribution without setting it"""
    hmm = deepcopy(hmm_lr)
    with pytest.raises(AttributeError) as e:
        hmm.initial
    assert str(e.value) == 'No initial state distribution has been defined'

# ========================== #
# HMM.transitions (property) #
# ========================== #

def test_transitions_without_setting():
    """Get transition matrix without setting it"""
    hmm = deepcopy(hmm_lr)
    with pytest.raises(AttributeError) as e:
        hmm.transitions
    assert str(e.value) == 'No transition matrix has been defined'

# ==================== #
# HMM.initial (setter) #
# ==================== #

def test_left_right_initial_left_right():
    """Set an initial state distribution generated by a left-right topology on a left-right HMM"""
    hmm = deepcopy(hmm_lr)
    topology = _LeftRightTopology(n_states=5, random_state=rng)
    initial = topology.random_initial()
    hmm.initial = initial
    assert_equal(hmm.initial, initial)

def test_left_right_initial_ergodic():
    """Set an initial state distribution generated by a left-right topology on an ergodic HMM"""
    hmm = deepcopy(hmm_lr)
    topology = _ErgodicTopology(n_states=5, random_state=rng)
    initial = topology.random_initial()
    hmm.initial = initial
    assert_equal(hmm.initial, initial)

def test_left_right_initial_strict_left_right():
    """Set an initial state distribution generated by a left-right topology on an strict left-right HMM"""
    hmm = deepcopy(hmm_lr)
    topology = _StrictLeftRightTopology(n_states=5, random_state=rng)
    initial = topology.random_initial()
    hmm.initial = initial
    assert_equal(hmm.initial, initial)

def test_ergodic_initial_left_right():
    """Set an initial state distribution generated by an ergodic topology on a left-right HMM"""
    hmm = deepcopy(hmm_e)
    topology = _LeftRightTopology(n_states=5, random_state=rng)
    initial = topology.random_initial()
    hmm.initial = initial
    assert_equal(hmm.initial, initial)

def test_ergodic_initial_ergodic():
    """Set an initial state distribution generated by an ergodic topology on an ergodic HMM"""
    hmm = deepcopy(hmm_e)
    topology = _ErgodicTopology(n_states=5, random_state=rng)
    initial = topology.random_initial()
    hmm.initial = initial
    assert_equal(hmm.initial, initial)

def test_ergodic_initial_strict_left_right():
    """Set an initial state distribution generated by an ergodic topology on a strict left-right HMM"""
    hmm = deepcopy(hmm_e)
    topology = _StrictLeftRightTopology(n_states=5, random_state=rng)
    initial = topology.random_initial()
    hmm.initial = initial
    assert_equal(hmm.initial, initial)

def test_strict_left_right_initial_left_right():
    """Set an initial state distribution generated by a strict left-right topology on a left-right HMM"""
    hmm = deepcopy(hmm_slr)
    topology = _LeftRightTopology(n_states=5, random_state=rng)
    initial = topology.random_initial()
    hmm.initial = initial
    assert_equal(hmm.initial, initial)

def test_strict_left_right_initial_ergodic():
    """Set an initial state distribution generated by a strict left-right topology on an ergodic HMM"""
    hmm = deepcopy(hmm_slr)
    topology = _ErgodicTopology(n_states=5, random_state=rng)
    initial = topology.random_initial()
    hmm.initial = initial
    assert_equal(hmm.initial, initial)

def test_strict_left_right_initial_strict_left_right():
    """Set an initial state distribution generated by a strict left-right topology on an strict left-right HMM"""
    hmm = deepcopy(hmm_slr)
    topology = _StrictLeftRightTopology(n_states=5, random_state=rng)
    initial = topology.random_initial()
    hmm.initial = initial
    assert_equal(hmm.initial, initial)

# ======================== #
# HMM.transitions (setter) #
# ======================== #

def test_left_right_transitions_left_right():
    """Set a transition matrix generated by a left-right topology on a left-right HMM"""
    hmm = deepcopy(hmm_lr)
    topology = _LeftRightTopology(n_states=5, random_state=rng)
    transitions = topology.random_transitions()
    hmm.transitions = transitions
    assert_equal(hmm.transitions, transitions)

def test_left_right_transitions_ergodic():
    """Set a transition matrix generated by a left-right topology on an ergodic HMM"""
    hmm = deepcopy(hmm_lr)
    topology = _ErgodicTopology(n_states=5, random_state=rng)
    transitions = topology.random_transitions()
    with pytest.raises(ValueError) as e:
        hmm.transitions = transitions
    assert str(e.value) == 'Left-right transition matrix must be upper-triangular'

def test_left_right_transitions_strict_left_right():
    """Set a transition matrix generated by a left-right topology on a strict left-right HMM"""
    hmm = deepcopy(hmm_lr)
    topology = _StrictLeftRightTopology(n_states=5, random_state=rng)
    transitions = topology.random_transitions()
    hmm.transitions = transitions
    assert_equal(hmm.transitions, transitions)

def test_ergodic_transitions_left_right():
    """Set a transition matrix generated by an ergodic topology on a left-right HMM"""
    hmm = deepcopy(hmm_e)
    topology = _LeftRightTopology(n_states=5, random_state=rng)
    transitions = topology.random_transitions()
    with pytest.warns(UserWarning) as w:
        hmm.transitions = transitions
    assert w[0].message.args[0] == 'Zero probabilities in ergodic transition matrix - these transition probabilities will not be learned'
    assert_equal(hmm.transitions, transitions)

def test_ergodic_transitions_ergodic():
    """Set a transition matrix generated by an ergodic topology on an ergodic HMM"""
    hmm = deepcopy(hmm_e)
    topology = _ErgodicTopology(n_states=5, random_state=rng)
    transitions = topology.random_transitions()
    hmm.transitions = transitions
    assert_equal(hmm.transitions, transitions)

def test_ergodic_transitions_strict_left_right():
    """Set a transition matrix generated by an ergodic topology on a strict left-right HMM"""
    hmm = deepcopy(hmm_e)
    topology = _StrictLeftRightTopology(n_states=5, random_state=rng)
    transitions = topology.random_transitions()
    with pytest.warns(UserWarning) as w:
        hmm.transitions = transitions
    assert w[0].message.args[0] == 'Zero probabilities in ergodic transition matrix - these transition probabilities will not be learned'
    assert_equal(hmm.transitions, transitions)

def test_strict_left_right_transitions_left_right():
    """Set a transition matrix generated by a strict left-right topology on a left-right HMM"""
    hmm = deepcopy(hmm_slr)
    topology = _LeftRightTopology(n_states=5, random_state=rng)
    transitions = topology.random_transitions()
    with pytest.raises(ValueError) as e:
        hmm.transitions = transitions
    assert str(e.value) == 'Strict left-right transition matrix must only consist of a diagonal and upper diagonal'

def test_strict_left_right_transitions_ergodic():
    """Set a transition matrix generated by a strict left-right topology on an ergodic HMM"""
    hmm = deepcopy(hmm_slr)
    topology = _ErgodicTopology(n_states=5, random_state=rng)
    transitions = topology.random_transitions()
    with pytest.raises(ValueError) as e:
        hmm.transitions = transitions
    assert str(e.value) == 'Left-right transition matrix must be upper-triangular'

def test_strict_left_right_transitions_strict_left_right():
    """Set a transition matrix generated by a strict left-right topology on a strict left-right HMM"""
    hmm = deepcopy(hmm_slr)
    topology = _StrictLeftRightTopology(n_states=5, random_state=rng)
    transitions = topology.random_transitions()
    hmm.transitions = transitions
    assert_equal(hmm.transitions, transitions)

# ============= #
# HMM.as_dict() #
# ============= #

def test_as_dict_unfitted():
    """Export an unfitted HMM to dict"""
    hmm = deepcopy(hmm_e)
    with pytest.raises(AttributeError) as e:
        hmm.as_dict()
    assert str(e.value) == 'The model needs to be fitted before it can be exported to a dict'

def test_as_dict_fitted():
    """Export a fitted HMM to dict"""
    hmm = deepcopy(hmm_e)
    hmm.set_uniform_initial()
    hmm.set_uniform_transitions()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        hmm.fit(X)
    d = hmm.as_dict()

    assert d['type'] == 'HMM'
    assert d['label'] == 'c1'
    assert d['n_states'] == 5
    assert d['topology'] == 'ergodic'
    assert_equal(d['model']['initial'], np.array([
        0.2, 0.2, 0.2, 0.2, 0.2
    ]))
    assert_equal(d['model']['transitions'], np.array([
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [0.2, 0.2, 0.2, 0.2, 0.2]
    ]))
    assert d['model']['n_seqs'] == 3
    assert d['model']['n_features'] == 3
    assert isinstance(d['model']['hmm'], dict)

# ========== #
# HMM.save() #
# ========== #

def test_save_directory():
    """Save a HMM into a directory"""
    hmm = deepcopy(hmm_e)
    hmm.set_uniform_initial()
    hmm.set_uniform_transitions()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        hmm.fit(X)
    with pytest.raises(IsADirectoryError) as e:
        hmm.save('.')
    assert str(e.value) == "[Errno 21] Is a directory: '.'"

def test_save_no_extension():
    """Save a HMM into a file without an extension"""
    try:
        hmm = deepcopy(hmm_e)
        hmm.set_uniform_initial()
        hmm.set_uniform_transitions()
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            hmm.fit(X)
        hmm.save('test')
        assert os.path.isfile('test')
    finally:
        os.remove('test')

def test_save_with_extension():
    """Save a HMM into a file with a .json extension"""
    try:
        hmm = deepcopy(hmm_e)
        hmm.set_uniform_initial()
        hmm.set_uniform_transitions()
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            hmm.fit(X)
        hmm.save('test.json')
        assert os.path.isfile('test.json')
    finally:
        os.remove('test.json')

# ========== #
# HMM.load() #
# ========== #

def test_load_invalid_dict():
    """Load a HMM from an invalid dict"""
    with pytest.raises(KeyError) as e:
        HMM.load({})

def test_load_dict():
    """Load a HMM from a valid dict"""
    hmm = deepcopy(hmm_lr)
    hmm.set_uniform_initial()
    hmm.set_uniform_transitions()
    before = hmm.initial, hmm.transitions
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        hmm.fit(X)
    try:
        hmm = HMM.load(hmm.as_dict())

        assert isinstance(hmm, HMM)
        assert hmm._label == 'c1'
        assert hmm._n_states == 5
        assert isinstance(hmm._topology, _LeftRightTopology)
        assert_not_equal(hmm._initial, before[0])
        assert_not_equal(hmm._transitions, before[1])
        assert hmm._n_seqs == 3
        assert hmm._n_features == 3
        assert isinstance(hmm._model, pg.HiddenMarkovModel)
    except:
        pass

def test_load_invalid_path():
    """Load a HMM from a directory"""
    with pytest.raises(IsADirectoryError) as e:
        HMM.load('.')

def test_load_inexistent_path():
    """Load a HMM from an inexistent path"""
    with pytest.raises(FileNotFoundError) as e:
        HMM.load('test')

def test_load_invalid_format():
    """Load a HMM from an illegally formatted file"""
    try:
        with open('test', 'w') as f:
            f.write('illegal')
        with pytest.raises(json.decoder.JSONDecodeError) as e:
            HMM.load('test')
    finally:
        os.remove('test')

def test_load_invalid_json():
    """Load a HMM from an invalid JSON file"""
    try:
        with open('test', 'w') as f:
            f.write("{}")
        with pytest.raises(KeyError) as e:
            HMM.load('test')
    finally:
        os.remove('test')

def test_load_path():
    """Load a HMM from a valid JSON file"""
    try:
        hmm = deepcopy(hmm_slr)
        hmm.set_uniform_initial()
        hmm.set_uniform_transitions()
        before = hmm.initial, hmm.transitions
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            hmm.fit(X)
        hmm.save('test')
        hmm = HMM.load('test')

        assert isinstance(hmm, HMM)
        assert hmm._label == 'c1'
        assert hmm._n_states == 5
        assert isinstance(hmm._topology, _StrictLeftRightTopology)
        assert_not_equal(hmm._initial, before[0])
        assert_not_equal(hmm._transitions, before[1])
        assert hmm._n_seqs == 3
        assert hmm._n_features == 3
        assert isinstance(hmm._model, pg.HiddenMarkovModel)
    finally:
        os.remove('test')