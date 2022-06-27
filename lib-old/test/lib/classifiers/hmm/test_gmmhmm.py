import pytest, warnings, os, math, numpy as np, hmmlearn.base, hmmlearn.hmm
from copy import deepcopy
from sequentia.classifiers import GMMHMM, _LeftRightTopology, _ErgodicTopology, _LinearTopology
from sequentia.datasets import load_random_sequences
from ....support import assert_equal, assert_not_equal, assert_all_equal, assert_all_not_equal

# Set seed for reproducible randomness
random_state = np.random.RandomState(0)

# Create some sample data
dataset = load_random_sequences(15, n_features=2, n_classes=2, length_range=(20, 30), random_state=random_state)
X = [x for x, y in dataset if y == 0]
x = X[0]

# Unparameterized HMMs
hmm_lr = GMMHMM(label='c1', n_states=5, topology='left-right', random_state=random_state)
hmm_e = GMMHMM(label='c1', n_states=5, topology='ergodic', random_state=random_state)
hmm_lin = GMMHMM(label='c1', n_states=5, topology='linear', random_state=random_state)

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
        0.53803322, 0.12404781, 0.07762362, 0.17663443, 0.08366092
    ]))

def test_ergodic_random_initial():
    """Random initial state distribution for an ergodic HMM"""
    hmm = deepcopy(hmm_e)
    hmm.set_random_initial()
    assert_equal(hmm.initial_, np.array([
        0.53803322, 0.12404781, 0.07762362, 0.17663443, 0.08366092
    ]))

def test_linear_random_initial():
    """Random initial state distribution for a linear HMM"""
    hmm = deepcopy(hmm_lin)
    hmm.set_random_initial()
    assert_equal(hmm.initial_, np.array([
        0.53803322, 0.12404781, 0.07762362, 0.17663443, 0.08366092
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
        [0.53803322, 0.12404781, 0.07762362, 0.17663443, 0.08366092],
        [0.        , 0.0544546 , 0.167254  , 0.43679272, 0.34149867],
        [0.        , 0.        , 0.02569653, 0.93686415, 0.03743932],
        [0.        , 0.        , 0.        , 0.80245882, 0.19754118],
        [0.        , 0.        , 0.        , 0.        , 1.        ]
    ]))

def test_ergodic_random_transitions():
    """Random transition matrix for an ergodic HMM"""
    hmm = deepcopy(hmm_e)
    hmm.set_random_transitions()
    assert_equal(hmm.transitions_, np.array([
        [0.53803322, 0.12404781, 0.07762362, 0.17663443, 0.08366092],
        [0.05407134, 0.16607684, 0.43371851, 0.33909515, 0.00703816],
        [0.12935118, 0.00516918, 0.67173292, 0.16536041, 0.02838631],
        [0.15768347, 0.31907791, 0.42873228, 0.06083948, 0.03366686],
        [0.42607069, 0.17697038, 0.33288653, 0.04212738, 0.02194502]
    ]))

def test_linear_random_transitions():
    """Random transition matrix for a linear HMM"""
    hmm = deepcopy(hmm_lin)
    hmm.set_random_transitions()
    assert_equal(hmm.transitions_, np.array([
        [0.81263954, 0.18736046, 0.        , 0.        , 0.        ],
        [0.        , 0.30529464, 0.69470536, 0.        , 0.        ],
        [0.        , 0.        , 0.34435856, 0.65564144, 0.        ],
        [0.        , 0.        , 0.        , 0.27688918, 0.72311082],
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
    assert hmm.n_seqs_ == 8
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
        0.53803322, 0.12404781, 0.07762362, 0.17663443, 0.08366092
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
        [0.05407134, 0.16607684, 0.43371851, 0.33909515, 0.00703816],
        [0.        , 0.13313025, 0.0053202 , 0.69135803, 0.17019152],
        [0.        , 0.        , 0.11443295, 0.29289135, 0.59267569],
        [0.        , 0.        , 0.        , 0.87572918, 0.12427082],
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
        0.53803322, 0.12404781, 0.07762362, 0.17663443, 0.08366092
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
        [0.05407134, 0.16607684, 0.43371851, 0.33909515, 0.00703816],
        [0.12935118, 0.00516918, 0.67173292, 0.16536041, 0.02838631],
        [0.15768347, 0.31907791, 0.42873228, 0.06083948, 0.03366686],
        [0.42607069, 0.17697038, 0.33288653, 0.04212738, 0.02194502],
        [0.20328414, 0.13729798, 0.03560389, 0.4874536 , 0.13636039]
    ]))
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
        0.53803322, 0.12404781, 0.07762362, 0.17663443, 0.08366092
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
    print(repr(before))
    assert_equal(before, np.array([
        [0.24561338, 0.75438662, 0.        , 0.        , 0.        ],
        [0.        , 0.56122003, 0.43877997, 0.        , 0.        ],
        [0.        , 0.        , 0.02669601, 0.97330399, 0.        ],
        [0.        , 0.        , 0.        , 0.00763653, 0.99236347],
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
    assert math.isclose(hmm.forward(x), -89.59052551245605, rel_tol=1e-8)

def test_ergodic_forward():
    """Forward algorithm on an ergodic HMM"""
    hmm = deepcopy(hmm_e)
    hmm.set_random_initial()
    hmm.set_random_transitions()
    hmm.fit(X)
    assert math.isclose(hmm.forward(x), -97.67911812603418, rel_tol=1e-8)

def test_linear_forward():
    """Forward algorithm on a linear HMM"""
    hmm = deepcopy(hmm_lin)
    hmm.set_random_initial()
    hmm.set_random_transitions()
    hmm.fit(X)
    assert math.isclose(hmm.forward(x), -90.25666060143605, rel_tol=1e-8)

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
    assert hmm.n_seqs_ == 8

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
        [[ 0.49517361,  0.79670013]],
        [[ 1.81277369, -2.45995611]],
        [[-0.61198527, -0.2621587 ]],
        [[ 1.40168717,  0.16718235]],
        [[-2.05338535, -3.13926956]]
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
        [[[ 1.38488559,  0.38570541],
          [ 0.38570541,  0.63293189]]],
        [[[ 1.73706667,  0.28568952],
          [ 0.28568952,  0.47176263]]],
        [[[ 0.99011246, -0.10938155],
          [-0.10938155,  0.01847633]]],
        [[[ 1.94877336, -0.17877102],
          [-0.17877102,  2.42880862]]],
        [[[ 3.52820275,  0.5571457 ],
          [ 0.5571457 ,  0.2716629 ]]]
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
    topology = _LeftRightTopology(n_states=5, random_state=random_state)
    initial = topology.random_initial()
    hmm.initial_ = initial
    assert_equal(hmm.initial_, initial)

def test_left_right_initial_ergodic():
    """Set an initial state distribution generated by a left-right topology on an ergodic HMM"""
    hmm = deepcopy(hmm_lr)
    topology = _ErgodicTopology(n_states=5, random_state=random_state)
    initial = topology.random_initial()
    hmm.initial_ = initial
    assert_equal(hmm.initial_, initial)

def test_left_right_initial_linear():
    """Set an initial state distribution generated by a left-right topology on an linear HMM"""
    hmm = deepcopy(hmm_lr)
    topology = _LinearTopology(n_states=5, random_state=random_state)
    initial = topology.random_initial()
    hmm.initial_ = initial
    assert_equal(hmm.initial_, initial)

def test_ergodic_initial_left_right():
    """Set an initial state distribution generated by an ergodic topology on a left-right HMM"""
    hmm = deepcopy(hmm_e)
    topology = _LeftRightTopology(n_states=5, random_state=random_state)
    initial = topology.random_initial()
    hmm.initial_ = initial
    assert_equal(hmm.initial_, initial)

def test_ergodic_initial_ergodic():
    """Set an initial state distribution generated by an ergodic topology on an ergodic HMM"""
    hmm = deepcopy(hmm_e)
    topology = _ErgodicTopology(n_states=5, random_state=random_state)
    initial = topology.random_initial()
    hmm.initial_ = initial
    assert_equal(hmm.initial_, initial)

def test_ergodic_initial_linear():
    """Set an initial state distribution generated by an ergodic topology on a linear HMM"""
    hmm = deepcopy(hmm_e)
    topology = _LinearTopology(n_states=5, random_state=random_state)
    initial = topology.random_initial()
    hmm.initial_ = initial
    assert_equal(hmm.initial_, initial)

def test_linear_initial_left_right():
    """Set an initial state distribution generated by a linear topology on a left-right HMM"""
    hmm = deepcopy(hmm_lin)
    topology = _LeftRightTopology(n_states=5, random_state=random_state)
    initial = topology.random_initial()
    hmm.initial_ = initial
    assert_equal(hmm.initial_, initial)

def test_linear_initial_ergodic():
    """Set an initial state distribution generated by a linear topology on an ergodic HMM"""
    hmm = deepcopy(hmm_lin)
    topology = _ErgodicTopology(n_states=5, random_state=random_state)
    initial = topology.random_initial()
    hmm.initial_ = initial
    assert_equal(hmm.initial_, initial)

def test_linear_initial_linear():
    """Set an initial state distribution generated by a linear topology on an linear HMM"""
    hmm = deepcopy(hmm_lin)
    topology = _LinearTopology(n_states=5, random_state=random_state)
    initial = topology.random_initial()
    hmm.initial_ = initial
    assert_equal(hmm.initial_, initial)

# ============================ #
# GMMHMM.transitions_ (setter) #
# ============================ #

def test_left_right_transitions_left_right():
    """Set a transition matrix generated by a left-right topology on a left-right HMM"""
    hmm = deepcopy(hmm_lr)
    topology = _LeftRightTopology(n_states=5, random_state=random_state)
    transitions = topology.random_transitions()
    hmm.transitions_ = transitions
    assert_equal(hmm.transitions_, transitions)

def test_left_right_transitions_ergodic():
    """Set a transition matrix generated by a left-right topology on an ergodic HMM"""
    hmm = deepcopy(hmm_lr)
    topology = _ErgodicTopology(n_states=5, random_state=random_state)
    transitions = topology.random_transitions()
    with pytest.raises(ValueError) as e:
        hmm.transitions_ = transitions
    assert str(e.value) == 'Left-right transition matrix must be upper-triangular'

def test_left_right_transitions_linear():
    """Set a transition matrix generated by a left-right topology on a linear HMM"""
    hmm = deepcopy(hmm_lr)
    topology = _LinearTopology(n_states=5, random_state=random_state)
    transitions = topology.random_transitions()
    hmm.transitions_ = transitions
    assert_equal(hmm.transitions_, transitions)

def test_ergodic_transitions_left_right():
    """Set a transition matrix generated by an ergodic topology on a left-right HMM"""
    hmm = deepcopy(hmm_e)
    topology = _LeftRightTopology(n_states=5, random_state=random_state)
    transitions = topology.random_transitions()
    with pytest.warns(UserWarning) as w:
        hmm.transitions_ = transitions
    assert w[0].message.args[0] == 'Zero probabilities in ergodic transition matrix - these transition probabilities will not be learned'
    assert_equal(hmm.transitions_, transitions)

def test_ergodic_transitions_ergodic():
    """Set a transition matrix generated by an ergodic topology on an ergodic HMM"""
    hmm = deepcopy(hmm_e)
    topology = _ErgodicTopology(n_states=5, random_state=random_state)
    transitions = topology.random_transitions()
    hmm.transitions_ = transitions
    assert_equal(hmm.transitions_, transitions)

def test_ergodic_transitions_linear():
    """Set a transition matrix generated by an ergodic topology on a linear HMM"""
    hmm = deepcopy(hmm_e)
    topology = _LinearTopology(n_states=5, random_state=random_state)
    transitions = topology.random_transitions()
    with pytest.warns(UserWarning) as w:
        hmm.transitions_ = transitions
    assert w[0].message.args[0] == 'Zero probabilities in ergodic transition matrix - these transition probabilities will not be learned'
    assert_equal(hmm.transitions_, transitions)

def test_linear_transitions_left_right():
    """Set a transition matrix generated by a linear topology on a left-right HMM"""
    hmm = deepcopy(hmm_lin)
    topology = _LeftRightTopology(n_states=5, random_state=random_state)
    transitions = topology.random_transitions()
    with pytest.raises(ValueError) as e:
        hmm.transitions_ = transitions
    assert str(e.value) == 'Linear transition matrix must only consist of a diagonal and upper diagonal'

def test_linear_transitions_ergodic():
    """Set a transition matrix generated by a linear topology on an ergodic HMM"""
    hmm = deepcopy(hmm_lin)
    topology = _ErgodicTopology(n_states=5, random_state=random_state)
    transitions = topology.random_transitions()
    with pytest.raises(ValueError) as e:
        hmm.transitions_ = transitions
    assert str(e.value) == 'Left-right transition matrix must be upper-triangular'

def test_linear_transitions_linear():
    """Set a transition matrix generated by a linear topology on a linear HMM"""
    hmm = deepcopy(hmm_lin)
    topology = _LinearTopology(n_states=5, random_state=random_state)
    transitions = topology.random_transitions()
    hmm.transitions_ = transitions
    assert_equal(hmm.transitions_, transitions)