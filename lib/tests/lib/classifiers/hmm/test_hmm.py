import pytest
import warnings
import numpy as np
from copy import deepcopy
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    import pomegranate as pg
from sequentia.classifiers import HMM, _LeftRightTopology, _ErgodicTopology
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

# ========================= #
# HMM.set_uniform_initial() #
# ========================= #

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

# ======================== #
# HMM.set_random_initial() #
# ======================== #

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

# ====================================================== #
# HMM.set_uniform_transitions() + HMM.initial (property) #
# ====================================================== #

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