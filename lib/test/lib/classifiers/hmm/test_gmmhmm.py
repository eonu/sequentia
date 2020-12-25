import pytest, warnings, os, numpy as np, hmmlearn.hmm
from copy import deepcopy
from sequentia.classifiers import GMMHMM, _LeftRightTopology, _ErgodicTopology, _LinearTopology
from ....support import assert_equal, assert_not_equal

# Set seed for reproducible randomness
seed = 0
np.random.seed(seed)
rng = np.random.RandomState(seed)

# Create some sample data
X = [rng.random((10 * i, 3)) for i in range(1, 4)]
x = rng.random((15, 3))

# Unparameterized HMMs
hmm_lr = GMMHMM(label='c1', n_states=5, topology='left-right', random_state=rng)
hmm_e = GMMHMM(label='c1', n_states=5, topology='ergodic', random_state=rng)
hmm_lin = GMMHMM(label='c1', n_states=5, topology='linear', random_state=rng)

# ======================================================== #
# GMMHMM.set_uniform_initial() + GMMHMM.initial (property) #
# ======================================================== #

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

def test_linear_uniform_initial():
    """Uniform initial state distribution for a linear HMM"""
    hmm = deepcopy(hmm_lin)
    hmm.set_uniform_initial()
    assert_equal(hmm.initial, np.array([
        0.2, 0.2, 0.2, 0.2, 0.2
    ]))

# ======================================================= #
# GMMHMM.set_random_initial() + GMMHMM.initial (property) #
# ======================================================= #

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

def test_linear_random_initial():
    """Random initial state distribution for a linear HMM"""
    hmm = deepcopy(hmm_lin)
    hmm.set_random_initial()
    assert_equal(hmm.initial, np.array([
        0.35029635, 0.13344569, 0.02784745, 0.33782453, 0.15058597
    ]))

# ================================================================ #
# GMMHMM.set_uniform_transitions() + GMMHMM.transitions (property) #
# ================================================================ #

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

def test_linear_uniform_transitions():
    """Uniform transition matrix for a linear HMM"""
    hmm = deepcopy(hmm_lin)
    hmm.set_uniform_transitions()
    assert_equal(hmm.transitions, np.array([
        [0.5, 0.5, 0. , 0. , 0. ],
        [0. , 0.5, 0.5, 0. , 0. ],
        [0. , 0. , 0.5, 0.5, 0. ],
        [0. , 0. , 0. , 0.5, 0.5],
        [0. , 0. , 0. , 0. , 1. ]
    ]))

# =============================================================== #
# GMMHMM.set_random_transitions() + GMMHMM.transitions (property) #
# =============================================================== #

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

def test_linear_random_transitions():
    """Random transition matrix for a linear HMM"""
    hmm = deepcopy(hmm_lin)
    hmm.set_random_transitions()
    assert_equal(hmm.transitions, np.array([
        [0.72413873, 0.27586127, 0.        , 0.        , 0.        ],
        [0.        , 0.07615418, 0.92384582, 0.        , 0.        ],
        [0.        , 0.        , 0.81752797, 0.18247203, 0.        ],
        [0.        , 0.        , 0.        , 0.24730529, 0.75269471],
        [0.        , 0.        , 0.        , 0.        , 1.        ]
    ]))

# ============================ #
# GMMHMM.fit() + GMMHMM.n_seqs #
# ============================ #

def test_fit_without_initial_and_transition():
    """Fitting before setting the initial state distribution and transition matrix"""
    hmm = deepcopy(hmm_lr)
    with pytest.raises(AttributeError) as e:
        hmm.fit(X)
    assert str(e.value) == 'Must specify initial state distribution and transitions before the HMM can be fitted'

def test_fit_without_transitions():
    """Fitting before setting the initial transition matrix"""
    hmm = deepcopy(hmm_lr)
    hmm.set_uniform_initial()
    with pytest.raises(AttributeError) as e:
        hmm.fit(X)
    assert str(e.value) == 'Must specify initial state distribution and transitions before the HMM can be fitted'

def test_fit_without_initial():
    """Fitting before setting the initial state distribution"""
    hmm = deepcopy(hmm_lr)
    hmm.set_uniform_transitions()
    with pytest.raises(AttributeError) as e:
        hmm.fit(X)
    assert str(e.value) == 'Must specify initial state distribution and transitions before the HMM can be fitted'

def test_fit_sets_internals():
    """Check that fitting sets internal attributes"""
    hmm = deepcopy(hmm_lr)
    hmm.set_uniform_initial()
    hmm.set_uniform_transitions()
    hmm.fit(X)
    assert hmm.n_seqs == 3
    assert isinstance(hmm._model, hmmlearn.hmm.GMMHMM)

def test_left_right_fit_updates_uniform_initial():
    """Check fitting with a uniform initial state distribution of a left-right HMM"""
    hmm = deepcopy(hmm_lr)
    hmm.set_uniform_initial()
    hmm.set_uniform_transitions()
    assert_equal(hmm.initial, np.array([
        0.2, 0.2, 0.2, 0.2, 0.2
    ]))
    before = hmm.initial
    hmm.fit(X)
    assert_not_equal(hmm.initial, before)

def test_left_right_fit_updates_random_initial():
    """Check fitting with a random initial state distribution of a left-right HMM"""
    hmm = deepcopy(hmm_lr)
    hmm.set_random_initial()
    hmm.set_random_transitions()
    assert_equal(hmm.initial, np.array([
        0.35029635, 0.13344569, 0.02784745, 0.33782453, 0.15058597
    ]))
    before = hmm.initial
    hmm.fit(X)
    assert_not_equal(hmm.initial, before)

def test_left_right_fit_updates_uniform_transitions():
    """Check fitting with a uniform transition matrix of a left-right HMM"""
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
    hmm.fit(X)
    assert_not_equal(hmm.transitions, before)

def test_left_right_fit_updates_random_transitions():
    """Check fitting with a random transition matrix of a left-right HMM"""
    hmm = deepcopy(hmm_lr)
    hmm.set_random_initial()
    hmm.set_random_transitions()
    before = hmm.transitions
    assert_equal(hmm.transitions, np.array([
        [0.19252534, 0.15767581, 0.47989976, 0.01708551, 0.15281357],
        [0.        , 0.21269278, 0.26671807, 0.16241481, 0.35817434],
        [0.        , 0.        , 0.33753566, 0.19947995, 0.46298439],
        [0.        , 0.        , 0.        , 0.39158159, 0.60841841],
        [0.        , 0.        , 0.        , 0.        , 1.        ]
    ]))
    hmm.fit(X)
    assert_not_equal(hmm.transitions, before)

def test_ergodic_fit_updates_uniform_initial():
    """Check fitting with a uniform initial state distribution of an ergodic HMM"""
    hmm = deepcopy(hmm_e)
    hmm.set_uniform_initial()
    hmm.set_uniform_transitions()
    assert_equal(hmm.initial, np.array([
        0.2, 0.2, 0.2, 0.2, 0.2
    ]))
    before = hmm.initial
    hmm.fit(X)
    assert_not_equal(hmm.initial, before)

def test_ergodic_fit_updates_random_initial():
    """Check fitting with a random initial state distribution of an ergodic HMM"""
    hmm = deepcopy(hmm_e)
    hmm.set_random_initial()
    hmm.set_random_transitions()
    assert_equal(hmm.initial, np.array([
        0.35029635, 0.13344569, 0.02784745, 0.33782453, 0.15058597
    ]))
    before = hmm.initial
    hmm.fit(X)
    assert_not_equal(hmm.initial, before)

def test_ergodic_fit_updates_uniform_transitions():
    """Check fitting with a uniform transition matrix of an ergodic HMM"""
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
    hmm.fit(X)
    assert_not_equal(hmm.transitions, before)

def test_ergodic_fit_updates_random_transitions():
    """Check fitting with a random transition matrix of an ergodic HMM"""
    hmm = deepcopy(hmm_e)
    hmm.set_random_initial()
    hmm.set_random_transitions()
    assert_equal(hmm.transitions, np.array([
        [0.19252534, 0.15767581, 0.47989976, 0.01708551, 0.15281357],
        [0.19375092, 0.16425506, 0.21828034, 0.11397708, 0.30973661],
        [0.46906977, 0.02941216, 0.17137502, 0.0333193 , 0.29682374],
        [0.21312406, 0.35221103, 0.08556524, 0.06613143, 0.28296824],
        [0.05212313, 0.3345513 , 0.17192948, 0.16379392, 0.27760217]]))
    before = hmm.transitions
    hmm.fit(X)
    assert_not_equal(hmm.transitions, before)

def test_linear_fit_updates_uniform_initial():
    """Check fitting with a uniform initial state distribution of a linear HMM"""
    hmm = deepcopy(hmm_lin)
    hmm.set_uniform_initial()
    hmm.set_uniform_transitions()
    assert_equal(hmm.initial, np.array([
        0.2, 0.2, 0.2, 0.2, 0.2
    ]))
    before = hmm.initial
    hmm.fit(X)
    assert_not_equal(hmm.initial, before)

def test_linear_fit_updates_random_initial():
    """Check fitting with a random initial state distribution of a linear HMM"""
    hmm = deepcopy(hmm_lin)
    hmm.set_random_initial()
    hmm.set_random_transitions()
    assert_equal(hmm.initial, np.array([
        0.35029635, 0.13344569, 0.02784745, 0.33782453, 0.15058597
    ]))
    before = hmm.initial
    hmm.fit(X)
    assert_not_equal(hmm.initial, before)

def test_linear_fit_updates_uniform_transitions():
    """Check fitting with a uniform transition matrix of a linear HMM"""
    hmm = deepcopy(hmm_lin)
    hmm.set_uniform_initial()
    hmm.set_uniform_transitions()
    before = hmm.transitions
    assert_equal(hmm.transitions, np.array([
        [0.5, 0.5, 0. , 0. , 0. ],
        [0. , 0.5, 0.5, 0. , 0. ],
        [0. , 0. , 0.5, 0.5, 0. ],
        [0. , 0. , 0. , 0.5, 0.5],
        [0. , 0. , 0. , 0. , 1. ]
    ]))
    hmm.fit(X)
    assert_not_equal(hmm.transitions, before)

def test_linear_fit_updates_random_transitions():
    """Check fitting with a random transition matrix of a linear HMM"""
    hmm = deepcopy(hmm_lin)
    hmm.set_random_initial()
    hmm.set_random_transitions()
    before = hmm.transitions
    assert_equal(hmm.transitions, np.array([
        [0.54975645, 0.45024355, 0.        , 0.        , 0.        ],
        [0.        , 0.96562169, 0.03437831, 0.        , 0.        ],
        [0.        , 0.        , 0.29607315, 0.70392685, 0.        ],
        [0.        , 0.        , 0.        , 0.42938524, 0.57061476],
        [0.        , 0.        , 0.        , 0.        , 1.        ]
    ]))
    hmm.fit(X)
    assert_not_equal(hmm.transitions, before)

# ================ #
# GMMHMM.forward() #
# ================ #

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
    hmm.fit(X)
    assert isinstance(hmm.forward(x), float)

def test_ergodic_forward():
    """Forward algorithm on an ergodic HMM"""
    hmm = deepcopy(hmm_e)
    hmm.set_random_initial()
    hmm.set_random_transitions()
    hmm.fit(X)
    assert isinstance(hmm.forward(x), float)

def test_linear_forward():
    """Forward algorithm on a linear HMM"""
    hmm = deepcopy(hmm_lin)
    hmm.set_random_initial()
    hmm.set_random_transitions()
    hmm.fit(X)
    assert isinstance(hmm.forward(x), float)

# ======================= #
# GMMHMM.label (property) #
# ======================= #

def test_label():
    assert deepcopy(hmm_lr).label == 'c1'

# ========================== #
# GMMHMM.n_states (property) #
# ========================== #

def test_n_states():
    assert deepcopy(hmm_lr).n_states == 5

# ============================== #
# GMMHMM.n_components (property) #
# ============================== #

def test_n_components():
    assert deepcopy(hmm_lr).n_components == 1

# ================================= #
# GMMHMM.covariance_type (property) #
# ================================= #

def test_covariance_type():
    assert deepcopy(hmm_lr).covariance_type == 'full'

# ======================== #
# GMMHMM.n_seqs (property) #
# ======================== #

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
    hmm.fit(X)
    assert hmm.n_seqs == 3

# ========================= #
# GMMHMM.initial (property) #
# ========================= #

def test_initial_without_setting():
    """Get initial state distribution without setting it"""
    hmm = deepcopy(hmm_lr)
    with pytest.raises(AttributeError) as e:
        hmm.initial
    assert str(e.value) == 'No initial state distribution has been defined'

# ============================= #
# GMMHMM.transitions (property) #
# ============================= #

def test_transitions_without_setting():
    """Get transition matrix without setting it"""
    hmm = deepcopy(hmm_lr)
    with pytest.raises(AttributeError) as e:
        hmm.transitions
    assert str(e.value) == 'No transition matrix has been defined'

# ======================= #
# GMMHMM.initial (setter) #
# ======================= #

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

def test_left_right_initial_linear():
    """Set an initial state distribution generated by a left-right topology on an linear HMM"""
    hmm = deepcopy(hmm_lr)
    topology = _LinearTopology(n_states=5, random_state=rng)
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

def test_ergodic_initial_linear():
    """Set an initial state distribution generated by an ergodic topology on a linear HMM"""
    hmm = deepcopy(hmm_e)
    topology = _LinearTopology(n_states=5, random_state=rng)
    initial = topology.random_initial()
    hmm.initial = initial
    assert_equal(hmm.initial, initial)

def test_linear_initial_left_right():
    """Set an initial state distribution generated by a linear topology on a left-right HMM"""
    hmm = deepcopy(hmm_lin)
    topology = _LeftRightTopology(n_states=5, random_state=rng)
    initial = topology.random_initial()
    hmm.initial = initial
    assert_equal(hmm.initial, initial)

def test_linear_initial_ergodic():
    """Set an initial state distribution generated by a linear topology on an ergodic HMM"""
    hmm = deepcopy(hmm_lin)
    topology = _ErgodicTopology(n_states=5, random_state=rng)
    initial = topology.random_initial()
    hmm.initial = initial
    assert_equal(hmm.initial, initial)

def test_linear_initial_linear():
    """Set an initial state distribution generated by a linear topology on an linear HMM"""
    hmm = deepcopy(hmm_lin)
    topology = _LinearTopology(n_states=5, random_state=rng)
    initial = topology.random_initial()
    hmm.initial = initial
    assert_equal(hmm.initial, initial)

# =========================== #
# GMMHMM.transitions (setter) #
# =========================== #

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

def test_left_right_transitions_linear():
    """Set a transition matrix generated by a left-right topology on a linear HMM"""
    hmm = deepcopy(hmm_lr)
    topology = _LinearTopology(n_states=5, random_state=rng)
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

def test_ergodic_transitions_linear():
    """Set a transition matrix generated by an ergodic topology on a linear HMM"""
    hmm = deepcopy(hmm_e)
    topology = _LinearTopology(n_states=5, random_state=rng)
    transitions = topology.random_transitions()
    with pytest.warns(UserWarning) as w:
        hmm.transitions = transitions
    assert w[0].message.args[0] == 'Zero probabilities in ergodic transition matrix - these transition probabilities will not be learned'
    assert_equal(hmm.transitions, transitions)

def test_linear_transitions_left_right():
    """Set a transition matrix generated by a linear topology on a left-right HMM"""
    hmm = deepcopy(hmm_lin)
    topology = _LeftRightTopology(n_states=5, random_state=rng)
    transitions = topology.random_transitions()
    with pytest.raises(ValueError) as e:
        hmm.transitions = transitions
    assert str(e.value) == 'Linear transition matrix must only consist of a diagonal and upper diagonal'

def test_linear_transitions_ergodic():
    """Set a transition matrix generated by a linear topology on an ergodic HMM"""
    hmm = deepcopy(hmm_lin)
    topology = _ErgodicTopology(n_states=5, random_state=rng)
    transitions = topology.random_transitions()
    with pytest.raises(ValueError) as e:
        hmm.transitions = transitions
    assert str(e.value) == 'Left-right transition matrix must be upper-triangular'

def test_linear_transitions_linear():
    """Set a transition matrix generated by a linear topology on a linear HMM"""
    hmm = deepcopy(hmm_lin)
    topology = _LinearTopology(n_states=5, random_state=rng)
    transitions = topology.random_transitions()
    hmm.transitions = transitions
    assert_equal(hmm.transitions, transitions)