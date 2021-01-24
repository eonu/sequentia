import pytest, warnings, os, numpy as np, hmmlearn.base, hmmlearn.hmm
from copy import deepcopy
from sequentia.classifiers import GMMHMM, _LeftRightTopology, _ErgodicTopology, _LinearTopology
from ....support import assert_equal, assert_not_equal, assert_all_equal, assert_all_not_equal

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

# ========================================================= #
# GMMHMM.set_uniform_initial() + GMMHMM.initial_ (property) #
# ========================================================= #

def test_left_right_uniform_initial():
    """Uniform initial state distribution for a left-right HMM"""
    hmm = deepcopy(hmm_lr)
    hmm.set_uniform_initial()
    assert_equal(hmm.initial_, np.array([
        0.2, 0.2, 0.2, 0.2, 0.2
    ]))

def test_ergodic_uniform_initial():
    """Uniform initial state distribution for an ergodic HMM"""
    hmm = deepcopy(hmm_e)
    hmm.set_uniform_initial()
    assert_equal(hmm.initial_, np.array([
        0.2, 0.2, 0.2, 0.2, 0.2
    ]))

def test_linear_uniform_initial():
    """Uniform initial state distribution for a linear HMM"""
    hmm = deepcopy(hmm_lin)
    hmm.set_uniform_initial()
    assert_equal(hmm.initial_, np.array([
        0.2, 0.2, 0.2, 0.2, 0.2
    ]))

# ======================================================== #
# GMMHMM.set_random_initial() + GMMHMM.initial_ (property) #
# ======================================================== #

def test_left_right_random_initial():
    """Random initial state distribution for a left-right HMM"""
    hmm = deepcopy(hmm_lr)
    hmm.set_random_initial()
    assert_equal(hmm.initial_, np.array([
        0.35029635, 0.13344569, 0.02784745, 0.33782453, 0.15058597
    ]))

def test_ergodic_random_initial():
    """Random initial state distribution for an ergodic HMM"""
    hmm = deepcopy(hmm_e)
    hmm.set_random_initial()
    assert_equal(hmm.initial_, np.array([
        0.35029635, 0.13344569, 0.02784745, 0.33782453, 0.15058597
    ]))

def test_linear_random_initial():
    """Random initial state distribution for a linear HMM"""
    hmm = deepcopy(hmm_lin)
    hmm.set_random_initial()
    assert_equal(hmm.initial_, np.array([
        0.35029635, 0.13344569, 0.02784745, 0.33782453, 0.15058597
    ]))

# ================================================================= #
# GMMHMM.set_uniform_transitions() + GMMHMM.transitions_ (property) #
# ================================================================= #

def test_left_right_uniform_transitions():
    """Uniform transition matrix for a left-right HMM"""
    hmm = deepcopy(hmm_lr)
    hmm.set_uniform_transitions()
    assert_equal(hmm.transitions_, np.array([
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
    assert_equal(hmm.transitions_, np.array([
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
    assert_equal(hmm.transitions_, np.array([
        [0.5, 0.5, 0. , 0. , 0. ],
        [0. , 0.5, 0.5, 0. , 0. ],
        [0. , 0. , 0.5, 0.5, 0. ],
        [0. , 0. , 0. , 0.5, 0.5],
        [0. , 0. , 0. , 0. , 1. ]
    ]))

# ================================================================ #
# GMMHMM.set_random_transitions() + GMMHMM.transitions_ (property) #
# ================================================================ #

def test_left_right_random_transitions():
    """Random transition matrix for a left-right HMM"""
    hmm = deepcopy(hmm_lr)
    hmm.set_random_transitions()
    assert_equal(hmm.transitions_, np.array([
        [0.35029635, 0.13344569, 0.02784745, 0.33782453, 0.15058597],
        [0.        , 0.22725263, 0.18611702, 0.56646299, 0.02016736],
        [0.        , 0.        , 0.18542075, 0.44084593, 0.37373332],
        [0.        , 0.        , 0.        , 0.65696153, 0.34303847],
        [0.        , 0.        , 0.        , 0.        , 1.        ]
    ]))

def test_ergodic_random_transitions():
    """Random transition matrix for an ergodic HMM"""
    hmm = deepcopy(hmm_e)
    hmm.set_random_transitions()
    assert_equal(hmm.transitions_, np.array([
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
    assert_equal(hmm.transitions_, np.array([
        [0.72413873, 0.27586127, 0.        , 0.        , 0.        ],
        [0.        , 0.07615418, 0.92384582, 0.        , 0.        ],
        [0.        , 0.        , 0.81752797, 0.18247203, 0.        ],
        [0.        , 0.        , 0.        , 0.24730529, 0.75269471],
        [0.        , 0.        , 0.        , 0.        , 1.        ]
    ]))

# ============================= #
# GMMHMM.fit() + GMMHMM.n_seqs_ #
# ============================= #

def test_fit_without_initial_and_transition():
    """Fitting before setting the initial state distribution and transition matrix"""
    hmm = deepcopy(hmm_lr)
    with pytest.raises(AttributeError) as e:
        hmm.fit(X)
    assert str(e.value) == 'No initial state distribution has been defined'

def test_fit_without_transitions():
    """Fitting before setting the initial transition matrix"""
    hmm = deepcopy(hmm_lr)
    hmm.set_uniform_initial()
    with pytest.raises(AttributeError) as e:
        hmm.fit(X)
    assert str(e.value) == 'No transition matrix has been defined'

def test_fit_without_initial():
    """Fitting before setting the initial state distribution"""
    hmm = deepcopy(hmm_lr)
    hmm.set_uniform_transitions()
    with pytest.raises(AttributeError) as e:
        hmm.fit(X)
    assert str(e.value) == 'No initial state distribution has been defined'

def test_fit_sets_internals():
    """Check that fitting sets internal attributes"""
    hmm = deepcopy(hmm_lr)
    hmm.set_uniform_initial()
    hmm.set_uniform_transitions()
    hmm.fit(X)
    assert hmm.n_seqs_ == 3
    assert isinstance(hmm.model, hmmlearn.hmm.GMMHMM)

def test_left_right_fit_updates_uniform_initial():
    """Check fitting with a uniform initial state distribution of a left-right HMM"""
    hmm = deepcopy(hmm_lr)
    hmm.set_uniform_initial()
    hmm.set_uniform_transitions()
    before = hmm.initial_
    assert_equal(before, np.array([
        0.2, 0.2, 0.2, 0.2, 0.2
    ]))
    hmm.fit(X)
    assert_not_equal(before, hmm.initial_)

def test_left_right_fit_updates_random_initial():
    """Check fitting with a random initial state distribution of a left-right HMM"""
    hmm = deepcopy(hmm_lr)
    hmm.set_random_initial()
    hmm.set_random_transitions()
    before = hmm.initial_
    assert_equal(before, np.array([
        0.35029635, 0.13344569, 0.02784745, 0.33782453, 0.15058597
    ]))
    hmm.fit(X)
    assert_not_equal(before, hmm.initial_)

def test_left_right_fit_updates_uniform_transitions():
    """Check fitting with a uniform transition matrix of a left-right HMM"""
    hmm = deepcopy(hmm_lr)
    hmm.set_uniform_initial()
    hmm.set_uniform_transitions()
    before = hmm.transitions_
    assert_equal(before, np.array([
        [0.2       , 0.2       , 0.2       , 0.2       , 0.2       ],
        [0.        , 0.25      , 0.25      , 0.25      , 0.25      ],
        [0.        , 0.        , 0.33333333, 0.33333333, 0.33333333],
        [0.        , 0.        , 0.        , 0.5       , 0.5       ],
        [0.        , 0.        , 0.        , 0.        , 1.        ]
    ]))
    hmm.fit(X)
    assert_not_equal(before, hmm.transitions_)

def test_left_right_fit_updates_random_transitions():
    """Check fitting with a random transition matrix of a left-right HMM"""
    hmm = deepcopy(hmm_lr)
    hmm.set_random_initial()
    hmm.set_random_transitions()
    before = hmm.transitions_
    assert_equal(before, np.array([
        [0.19252534, 0.15767581, 0.47989976, 0.01708551, 0.15281357],
        [0.        , 0.28069128, 0.23795997, 0.31622761, 0.16512114],
        [0.        , 0.        , 0.29431489, 0.66404724, 0.04163787],
        [0.        , 0.        , 0.        , 0.8372241 , 0.1627759 ],
        [0.        , 0.        , 0.        , 0.        , 1.        ]
    ]))
    hmm.fit(X)
    assert_not_equal(before, hmm.transitions_)

def test_ergodic_fit_updates_uniform_initial():
    """Check fitting with a uniform initial state distribution of an ergodic HMM"""
    hmm = deepcopy(hmm_e)
    hmm.set_uniform_initial()
    hmm.set_uniform_transitions()
    before = hmm.initial_
    assert_equal(before, np.array([
        0.2, 0.2, 0.2, 0.2, 0.2
    ]))
    hmm.fit(X)
    assert_not_equal(before, hmm.initial_)

def test_ergodic_fit_updates_random_initial():
    """Check fitting with a random initial state distribution of an ergodic HMM"""
    hmm = deepcopy(hmm_e)
    hmm.set_random_initial()
    hmm.set_random_transitions()
    before = hmm.initial_
    assert_equal(before, np.array([
        0.35029635, 0.13344569, 0.02784745, 0.33782453, 0.15058597
    ]))
    hmm.fit(X)
    assert_not_equal(before, hmm.initial_)

def test_ergodic_fit_updates_uniform_transitions():
    """Check fitting with a uniform transition matrix of an ergodic HMM"""
    hmm = deepcopy(hmm_e)
    hmm.set_uniform_initial()
    hmm.set_uniform_transitions()
    before = hmm.transitions_
    assert_equal(before, np.array([
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [0.2, 0.2, 0.2, 0.2, 0.2]
    ]))
    hmm.fit(X)
    assert_not_equal(before, hmm.transitions_)

def test_ergodic_fit_updates_random_transitions():
    """Check fitting with a random transition matrix of an ergodic HMM"""
    hmm = deepcopy(hmm_e)
    hmm.set_random_initial()
    hmm.set_random_transitions()
    before = hmm.transitions_
    assert_equal(before, np.array([
        [0.19252534, 0.15767581, 0.47989976, 0.01708551, 0.15281357],
        [0.19375092, 0.16425506, 0.21828034, 0.11397708, 0.30973661],
        [0.46906977, 0.02941216, 0.17137502, 0.0333193 , 0.29682374],
        [0.21312406, 0.35221103, 0.08556524, 0.06613143, 0.28296824],
        [0.05212313, 0.3345513 , 0.17192948, 0.16379392, 0.27760217]]))
    hmm.fit(X)
    assert_not_equal(before, hmm.transitions_)

def test_linear_fit_updates_uniform_initial():
    """Check fitting with a uniform initial state distribution of a linear HMM"""
    hmm = deepcopy(hmm_lin)
    hmm.set_uniform_initial()
    hmm.set_uniform_transitions()
    before = hmm.initial_
    assert_equal(before, np.array([
        0.2, 0.2, 0.2, 0.2, 0.2
    ]))
    hmm.fit(X)
    assert_not_equal(before, hmm.initial_)

def test_linear_fit_updates_random_initial():
    """Check fitting with a random initial state distribution of a linear HMM"""
    hmm = deepcopy(hmm_lin)
    hmm.set_random_initial()
    hmm.set_random_transitions()
    before = hmm.initial_
    assert_equal(before, np.array([
        0.35029635, 0.13344569, 0.02784745, 0.33782453, 0.15058597
    ]))
    hmm.fit(X)
    assert_not_equal(before, hmm.initial_)

def test_linear_fit_updates_uniform_transitions():
    """Check fitting with a uniform transition matrix of a linear HMM"""
    hmm = deepcopy(hmm_lin)
    hmm.set_uniform_initial()
    hmm.set_uniform_transitions()
    before = hmm.transitions_
    assert_equal(before, np.array([
        [0.5, 0.5, 0. , 0. , 0. ],
        [0. , 0.5, 0.5, 0. , 0. ],
        [0. , 0. , 0.5, 0.5, 0. ],
        [0. , 0. , 0. , 0.5, 0.5],
        [0. , 0. , 0. , 0. , 1. ]
    ]))
    hmm.fit(X)
    assert_not_equal(before, hmm.transitions_)

def test_linear_fit_updates_random_transitions():
    """Check fitting with a random transition matrix of a linear HMM"""
    hmm = deepcopy(hmm_lin)
    hmm.set_random_initial()
    hmm.set_random_transitions()
    before = hmm.transitions_
    assert_equal(before, np.array([
        [0.54975645, 0.45024355, 0.        , 0.        , 0.        ],
        [0.        , 0.96562169, 0.03437831, 0.        , 0.        ],
        [0.        , 0.        , 0.29607315, 0.70392685, 0.        ],
        [0.        , 0.        , 0.        , 0.42938524, 0.57061476],
        [0.        , 0.        , 0.        , 0.        , 1.        ]
    ]))
    hmm.fit(X)
    assert_not_equal(before, hmm.transitions_)

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
    assert str(e.value) == 'The model must be fitted first'

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

# =============== #
# GMMHMM.freeze() #
# =============== #

def test_freeze_no_params():
    hmm = deepcopy(hmm_lin)
    hmm.set_random_initial()
    hmm.set_random_transitions()
    hmm.freeze()
    assert hmm.frozen == set('stmcw')
    before = (hmm.initial_, hmm.transitions_)
    hmm.fit(X)
    assert_all_equal(before, (hmm.initial_, hmm.transitions_))

def test_freeze_invalid_params_type():
    hmm = deepcopy(hmm_lin)
    hmm.set_random_initial()
    hmm.set_random_transitions()
    with pytest.raises(TypeError) as e:
        hmm.freeze(0)
    assert str(e.value) == "Expected a string consisting of any combination of 's', 't', 'm', 'c', 'w'"

def test_freeze_invalid_params_content():
    hmm = deepcopy(hmm_lin)
    hmm.set_random_initial()
    hmm.set_random_transitions()
    with pytest.raises(ValueError) as e:
        hmm.freeze('stmcwz')
    assert str(e.value) == "Expected a string consisting of any combination of 's', 't', 'm', 'c', 'w'"

def test_freeze_valid_params():
    hmm = deepcopy(hmm_lin)
    hmm.set_random_initial()
    hmm.set_random_transitions()
    hmm.freeze('sw')
    assert hmm.frozen == set('sw')
    initial_before, transitions_before = hmm.initial_, hmm.transitions_
    hmm.fit(X)
    assert_equal(initial_before, hmm.initial_)
    assert_not_equal(transitions_before, hmm.transitions_)

def test_freeze_duplicate_params():
    hmm = deepcopy(hmm_lin)
    hmm.set_random_initial()
    hmm.set_random_transitions()
    hmm.freeze('swsswwssswww')
    assert hmm.frozen == set('sw')
    initial_before, transitions_before = hmm.initial_, hmm.transitions_
    hmm.fit(X)
    assert_equal(initial_before, hmm.initial_)
    assert_not_equal(transitions_before, hmm.transitions_)

def test_freeze_all_params():
    hmm = deepcopy(hmm_lin)
    hmm.set_random_initial()
    hmm.set_random_transitions()
    hmm.freeze('stmcw')
    assert hmm.frozen == set('stmcw')
    before = (hmm.initial_, hmm.transitions_)
    hmm.fit(X)
    assert_all_equal(before, (hmm.initial_, hmm.transitions_))

# ================= #
# GMMHMM.unfreeze() #
# ================= #

def test_unfreeze_no_params():
    hmm = deepcopy(hmm_lin)
    hmm.set_random_initial()
    hmm.set_random_transitions()
    hmm.freeze()
    assert hmm.frozen == set('stmcw')
    hmm.unfreeze()
    assert hmm.frozen == set()
    before = (hmm.initial_, hmm.transitions_)
    hmm.fit(X)
    assert_all_not_equal(before, (hmm.initial_, hmm.transitions_))

def test_unfreeze_invalid_params_type():
    hmm = deepcopy(hmm_lin)
    hmm.set_random_initial()
    hmm.set_random_transitions()
    hmm.freeze()
    with pytest.raises(TypeError) as e:
        hmm.unfreeze(0)
    assert str(e.value) == "Expected a string consisting of any combination of 's', 't', 'm', 'c', 'w'"

def test_unfreeze_invalid_params_content():
    hmm = deepcopy(hmm_lin)
    hmm.set_random_initial()
    hmm.set_random_transitions()
    hmm.freeze()
    with pytest.raises(ValueError) as e:
        hmm.unfreeze('stmcwz')
    assert str(e.value) == "Expected a string consisting of any combination of 's', 't', 'm', 'c', 'w'"

def test_unfreeze_valid_params():
    hmm = deepcopy(hmm_lin)
    hmm.set_random_initial()
    hmm.set_random_transitions()
    hmm.freeze('st')
    hmm.unfreeze('t')
    assert hmm.frozen == set('s')
    initial_before, transitions_before = hmm.initial_, hmm.transitions_
    hmm.fit(X)
    assert_equal(initial_before, hmm.initial_)
    assert_not_equal(transitions_before, hmm.transitions_)

def test_unfreeze_duplicate_params():
    hmm = deepcopy(hmm_lin)
    hmm.set_random_initial()
    hmm.set_random_transitions()
    hmm.freeze('st')
    hmm.unfreeze('tttmwcmwct')
    assert hmm.frozen == set('s')
    initial_before, transitions_before = hmm.initial_, hmm.transitions_
    hmm.fit(X)
    assert_equal(initial_before, hmm.initial_)
    assert_not_equal(transitions_before, hmm.transitions_)

def test_unfreeze_all_params():
    hmm = deepcopy(hmm_lin)
    hmm.set_random_initial()
    hmm.set_random_transitions()
    hmm.freeze()
    assert hmm.frozen == set('stmcw')
    hmm.unfreeze()
    assert hmm.frozen == set()
    before = (hmm.initial_, hmm.transitions_)
    hmm.fit(X)
    assert_all_not_equal(before, (hmm.initial_, hmm.transitions_))

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

# ========================= #
# GMMHMM.n_seqs_ (property) #
# ========================= #

def test_n_seqs_without_fit():
    """Number of sequences without fitting the HMM"""
    hmm = deepcopy(hmm_lr)
    hmm.set_random_initial()
    hmm.set_random_transitions()
    with pytest.raises(AttributeError) as e:
        hmm.n_seqs_
    assert str(e.value) == 'The model has not been fitted and has not seen any observation sequences'

def test_n_seqs_with_fit():
    """Number of sequences after fitting the HMM"""
    hmm = deepcopy(hmm_lr)
    hmm.set_random_initial()
    hmm.set_random_transitions()
    hmm.fit(X)
    assert hmm.n_seqs_ == 3

# ======================== #
# GMMHMM.frozen (property) #
# ======================== #

def test_frozen():
    hmm = deepcopy(hmm_lr)
    hmm.freeze('sw')
    assert hmm.frozen == set('sw')

# ========================== #
# GMMHMM.monitor_ (property) #
# ========================== #

def test_monitor_without_fit():
    """Convergence monitor without fitting the HMM"""
    hmm = deepcopy(hmm_lr)
    hmm.set_random_initial()
    hmm.set_random_transitions()
    with pytest.raises(AttributeError) as e:
        hmm.monitor_
    assert str(e.value) == 'The model must be fitted first'

def test_monitor_with_fit():
    """Convergence monitor after fitting the HMM"""
    hmm = deepcopy(hmm_lr)
    hmm.set_random_initial()
    hmm.set_random_transitions()
    hmm.fit(X)
    assert isinstance(hmm.monitor_, hmmlearn.base.ConvergenceMonitor)

# ========================== #
# GMMHMM.weights_ (property) #
# ========================== #

def test_weights_without_fit():
    """GMM weights without fitting the HMM"""
    hmm = deepcopy(hmm_lr)
    hmm.set_random_initial()
    hmm.set_random_transitions()
    with pytest.raises(AttributeError) as e:
        hmm.weights_
    assert str(e.value) == 'The model must be fitted first'

def test_weights_with_fit():
    """GMM weights after fitting the HMM"""
    hmm = deepcopy(hmm_lr)
    hmm.set_random_initial()
    hmm.set_random_transitions()
    hmm.fit(X)
    assert_equal(hmm.weights_, np.array([
        [1.],
        [1.],
        [1.],
        [1.],
        [1.]
    ]))

# ======================== #
# GMMHMM.means_ (property) #
# ======================== #

def test_means_without_fit():
    """GMM mean vectors without fitting the HMM"""
    hmm = deepcopy(hmm_lr)
    hmm.set_random_initial()
    hmm.set_random_transitions()
    with pytest.raises(AttributeError) as e:
        hmm.means_
    assert str(e.value) == 'The model must be fitted first'

def test_means_with_fit():
    """GMM mean vectors after fitting the HMM"""
    hmm = deepcopy(hmm_lr)
    hmm.set_random_initial()
    hmm.set_random_transitions()
    hmm.fit(X)
    assert_equal(hmm.means_, np.array([
        [[0.31874666, 0.66724147, 0.13182087]],
        [[0.31856896, 0.66741038, 0.13179786]],
        [[0.71632403, 0.28939952, 0.18320713]],
        [[0.51787902, 0.57561888, 0.5995548 ]],
        [[0.66975947, 0.26867588, 0.25477769]]
    ]))

# ========================= #
# GMMHMM.covars_ (property) #
# ========================= #

def test_covars_without_fit():
    """GMM covariance matrices without fitting the HMM"""
    hmm = deepcopy(hmm_lr)
    hmm.set_random_initial()
    hmm.set_random_transitions()
    with pytest.raises(AttributeError) as e:
        hmm.covars_
    assert str(e.value) == 'The model must be fitted first'

def test_covars_with_fit():
    """GMM covariance matrices after fitting the HMM"""
    hmm = deepcopy(hmm_lr)
    hmm.set_random_initial()
    hmm.set_random_transitions()
    hmm.fit(X)
    assert_equal(hmm.covars_, np.array([
        [[[ 0.08307002,  0.00160875,  0.0157381 ],
          [ 0.00160875,  0.08735411, -0.01063379],
          [ 0.0157381 , -0.01063379,  0.08286247]]],
        [[[ 0.08307002,  0.00160875,  0.0157381 ],
          [ 0.00160875,  0.08735411, -0.01063379],
          [ 0.0157381 , -0.01063379,  0.08286247]]],
        [[[ 0.08307002,  0.00160875,  0.0157381 ],
          [ 0.00160875,  0.08735411, -0.01063379],
          [ 0.0157381 , -0.01063379,  0.08286247]]],
        [[[ 0.08307002,  0.00160875,  0.0157381 ],
          [ 0.00160875,  0.08735411, -0.01063379],
          [ 0.0157381 , -0.01063379,  0.08286247]]],
        [[[ 0.08307002,  0.00160875,  0.0157381 ],
          [ 0.00160875,  0.08735411, -0.01063379],
          [ 0.0157381 , -0.01063379,  0.08286247]]]
    ]))

# ========================== #
# GMMHMM.initial_ (property) #
# ========================== #

def test_initial_without_setting():
    """Get initial state distribution without setting it"""
    hmm = deepcopy(hmm_lr)
    with pytest.raises(AttributeError) as e:
        hmm.initial_
    assert str(e.value) == 'No initial state distribution has been defined'

# ============================== #
# GMMHMM.transitions_ (property) #
# ============================== #

def test_transitions_without_setting():
    """Get transition matrix without setting it"""
    hmm = deepcopy(hmm_lr)
    with pytest.raises(AttributeError) as e:
        hmm.transitions_
    assert str(e.value) == 'No transition matrix has been defined'

# ======================== #
# GMMHMM.initial_ (setter) #
# ======================== #

def test_left_right_initial_left_right():
    """Set an initial state distribution generated by a left-right topology on a left-right HMM"""
    hmm = deepcopy(hmm_lr)
    topology = _LeftRightTopology(n_states=5, random_state=rng)
    initial = topology.random_initial()
    hmm.initial_ = initial
    assert_equal(hmm.initial_, initial)

def test_left_right_initial_ergodic():
    """Set an initial state distribution generated by a left-right topology on an ergodic HMM"""
    hmm = deepcopy(hmm_lr)
    topology = _ErgodicTopology(n_states=5, random_state=rng)
    initial = topology.random_initial()
    hmm.initial_ = initial
    assert_equal(hmm.initial_, initial)

def test_left_right_initial_linear():
    """Set an initial state distribution generated by a left-right topology on an linear HMM"""
    hmm = deepcopy(hmm_lr)
    topology = _LinearTopology(n_states=5, random_state=rng)
    initial = topology.random_initial()
    hmm.initial_ = initial
    assert_equal(hmm.initial_, initial)

def test_ergodic_initial_left_right():
    """Set an initial state distribution generated by an ergodic topology on a left-right HMM"""
    hmm = deepcopy(hmm_e)
    topology = _LeftRightTopology(n_states=5, random_state=rng)
    initial = topology.random_initial()
    hmm.initial_ = initial
    assert_equal(hmm.initial_, initial)

def test_ergodic_initial_ergodic():
    """Set an initial state distribution generated by an ergodic topology on an ergodic HMM"""
    hmm = deepcopy(hmm_e)
    topology = _ErgodicTopology(n_states=5, random_state=rng)
    initial = topology.random_initial()
    hmm.initial_ = initial
    assert_equal(hmm.initial_, initial)

def test_ergodic_initial_linear():
    """Set an initial state distribution generated by an ergodic topology on a linear HMM"""
    hmm = deepcopy(hmm_e)
    topology = _LinearTopology(n_states=5, random_state=rng)
    initial = topology.random_initial()
    hmm.initial_ = initial
    assert_equal(hmm.initial_, initial)

def test_linear_initial_left_right():
    """Set an initial state distribution generated by a linear topology on a left-right HMM"""
    hmm = deepcopy(hmm_lin)
    topology = _LeftRightTopology(n_states=5, random_state=rng)
    initial = topology.random_initial()
    hmm.initial_ = initial
    assert_equal(hmm.initial_, initial)

def test_linear_initial_ergodic():
    """Set an initial state distribution generated by a linear topology on an ergodic HMM"""
    hmm = deepcopy(hmm_lin)
    topology = _ErgodicTopology(n_states=5, random_state=rng)
    initial = topology.random_initial()
    hmm.initial_ = initial
    assert_equal(hmm.initial_, initial)

def test_linear_initial_linear():
    """Set an initial state distribution generated by a linear topology on an linear HMM"""
    hmm = deepcopy(hmm_lin)
    topology = _LinearTopology(n_states=5, random_state=rng)
    initial = topology.random_initial()
    hmm.initial_ = initial
    assert_equal(hmm.initial_, initial)

# ============================ #
# GMMHMM.transitions_ (setter) #
# ============================ #

def test_left_right_transitions_left_right():
    """Set a transition matrix generated by a left-right topology on a left-right HMM"""
    hmm = deepcopy(hmm_lr)
    topology = _LeftRightTopology(n_states=5, random_state=rng)
    transitions = topology.random_transitions()
    hmm.transitions_ = transitions
    assert_equal(hmm.transitions_, transitions)

def test_left_right_transitions_ergodic():
    """Set a transition matrix generated by a left-right topology on an ergodic HMM"""
    hmm = deepcopy(hmm_lr)
    topology = _ErgodicTopology(n_states=5, random_state=rng)
    transitions = topology.random_transitions()
    with pytest.raises(ValueError) as e:
        hmm.transitions_ = transitions
    assert str(e.value) == 'Left-right transition matrix must be upper-triangular'

def test_left_right_transitions_linear():
    """Set a transition matrix generated by a left-right topology on a linear HMM"""
    hmm = deepcopy(hmm_lr)
    topology = _LinearTopology(n_states=5, random_state=rng)
    transitions = topology.random_transitions()
    hmm.transitions_ = transitions
    assert_equal(hmm.transitions_, transitions)

def test_ergodic_transitions_left_right():
    """Set a transition matrix generated by an ergodic topology on a left-right HMM"""
    hmm = deepcopy(hmm_e)
    topology = _LeftRightTopology(n_states=5, random_state=rng)
    transitions = topology.random_transitions()
    with pytest.warns(UserWarning) as w:
        hmm.transitions_ = transitions
    assert w[0].message.args[0] == 'Zero probabilities in ergodic transition matrix - these transition probabilities will not be learned'
    assert_equal(hmm.transitions_, transitions)

def test_ergodic_transitions_ergodic():
    """Set a transition matrix generated by an ergodic topology on an ergodic HMM"""
    hmm = deepcopy(hmm_e)
    topology = _ErgodicTopology(n_states=5, random_state=rng)
    transitions = topology.random_transitions()
    hmm.transitions_ = transitions
    assert_equal(hmm.transitions_, transitions)

def test_ergodic_transitions_linear():
    """Set a transition matrix generated by an ergodic topology on a linear HMM"""
    hmm = deepcopy(hmm_e)
    topology = _LinearTopology(n_states=5, random_state=rng)
    transitions = topology.random_transitions()
    with pytest.warns(UserWarning) as w:
        hmm.transitions_ = transitions
    assert w[0].message.args[0] == 'Zero probabilities in ergodic transition matrix - these transition probabilities will not be learned'
    assert_equal(hmm.transitions_, transitions)

def test_linear_transitions_left_right():
    """Set a transition matrix generated by a linear topology on a left-right HMM"""
    hmm = deepcopy(hmm_lin)
    topology = _LeftRightTopology(n_states=5, random_state=rng)
    transitions = topology.random_transitions()
    with pytest.raises(ValueError) as e:
        hmm.transitions_ = transitions
    assert str(e.value) == 'Linear transition matrix must only consist of a diagonal and upper diagonal'

def test_linear_transitions_ergodic():
    """Set a transition matrix generated by a linear topology on an ergodic HMM"""
    hmm = deepcopy(hmm_lin)
    topology = _ErgodicTopology(n_states=5, random_state=rng)
    transitions = topology.random_transitions()
    with pytest.raises(ValueError) as e:
        hmm.transitions_ = transitions
    assert str(e.value) == 'Left-right transition matrix must be upper-triangular'

def test_linear_transitions_linear():
    """Set a transition matrix generated by a linear topology on a linear HMM"""
    hmm = deepcopy(hmm_lin)
    topology = _LinearTopology(n_states=5, random_state=rng)
    transitions = topology.random_transitions()
    hmm.transitions_ = transitions
    assert_equal(hmm.transitions_, transitions)