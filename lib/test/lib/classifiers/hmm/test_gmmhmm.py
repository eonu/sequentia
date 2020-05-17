import pytest, warnings, os, json, numpy as np
from copy import deepcopy
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    import pomegranate as pg
from sequentia.classifiers import HMM, GMMHMM, _LeftRightTopology, _ErgodicTopology, _StrictLeftRightTopology
from ....support import assert_equal, assert_not_equal

# Set seed for reproducible randomness
seed = 0
np.random.seed(seed)
rng = np.random.RandomState(seed)

# Create some sample data
X = [rng.random((10 * i, 3)) for i in range(1, 4)]
x = rng.random((15, 3))

# Unparameterized HMMs
hmm_diag = GMMHMM(label='c1', n_states=5, n_components=5, covariance='diagonal', random_state=rng)
hmm_full = GMMHMM(label='c1', n_states=5, n_components=5, covariance='full', random_state=rng)

# ============================== #
# GMMHMM.n_components (property) #
# ============================== #

def test_n_components():
    assert deepcopy(hmm_diag).n_components == 5

# ============================ #
# GMMHMM.covariance (property) #
# ============================ #

def test_covariance():
    assert deepcopy(hmm_diag).covariance == 'diagonal'

# ================ #
# GMMHMM.as_dict() #
# ================ #

def test_as_dict_unfitted():
    """Export an unfitted GMMHMM to dict"""
    hmm = deepcopy(hmm_diag)
    with pytest.raises(AttributeError) as e:
        hmm.as_dict()
    assert str(e.value) == 'The model needs to be fitted before it can be exported to a dict'

def test_as_dict_fitted():
    """Export a fitted GMMHMM to dict"""
    hmm = deepcopy(hmm_diag)
    hmm.set_uniform_initial()
    hmm.set_uniform_transitions()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        hmm.fit(X)
    d = hmm.as_dict()

    assert d['type'] == 'GMMHMM'
    assert d['label'] == 'c1'
    assert d['n_states'] == 5
    assert d['n_components'] == 5
    assert d['covariance'] == 'diagonal'
    assert d['topology'] == 'left-right'
    assert_equal(d['model']['initial'], np.array([
        1.0, 0.0, 1.5900681624889484e-35, 4.496137626463639e-37, 7.327577727627556e-35
    ]))
    assert_equal(d['model']['transitions'], np.array([
        [0.00000000e+00, 6.66666667e-01, 4.68887399e-07, 3.33332730e-01, 1.34734441e-07],
        [0.00000000e+00, 0.00000000e+00, 9.99999875e-01, 2.43928881e-08, 1.00588498e-07],
        [0.00000000e+00, 0.00000000e+00, 3.22840529e-02, 9.35801370e-01, 3.19145768e-02],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 5.18319902e-01, 4.81680098e-01],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    ]))
    assert d['model']['n_seqs'] == 3
    assert d['model']['n_features'] == 3
    assert isinstance(d['model']['hmm'], dict)

# ========== #
# HMM.save() #
# ========== #

def test_save_directory():
    """Save a GMMHMM into a directory"""
    hmm = deepcopy(hmm_diag)
    hmm.set_uniform_initial()
    hmm.set_uniform_transitions()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        hmm.fit(X)
    with pytest.raises(IsADirectoryError) as e:
        hmm.save('.')
    assert str(e.value) == "[Errno 21] Is a directory: '.'"

def test_save_no_extension():
    """Save a GMMHMM into a file without an extension"""
    try:
        hmm = deepcopy(hmm_diag)
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
    """Save a GMMHMM into a file with a .json extension"""
    try:
        hmm = deepcopy(hmm_diag)
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
    """Load a GMMHMM from an invalid dict"""
    with pytest.raises(KeyError) as e:
        GMMHMM.load({})

def test_load_dict():
    """Load a GMMHMM from a valid dict"""
    hmm = deepcopy(hmm_diag)
    hmm.set_uniform_initial()
    hmm.set_uniform_transitions()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        hmm.fit(X)
    hmm = GMMHMM.load(hmm.as_dict())

    assert isinstance(hmm, GMMHMM)
    assert hmm._label == 'c1'
    assert hmm._n_states == 5
    assert hmm._n_components == 5
    assert hmm._covariance == 'diagonal'
    assert isinstance(hmm._topology, _LeftRightTopology)
    assert_equal(hmm._initial, np.array([
        1.00000000e+00, 0.00000000e+00, 1.59006816e-35, 4.49613763e-37, 7.32757773e-35
    ]))
    assert_equal(hmm._transitions, np.array([
        [0.00000000e+00, 6.66666667e-01, 4.68887399e-07, 3.33332730e-01, 1.34734441e-07],
        [0.00000000e+00, 0.00000000e+00, 9.99999875e-01, 2.43928881e-08, 1.00588498e-07],
        [0.00000000e+00, 0.00000000e+00, 3.22840529e-02, 9.35801370e-01, 3.19145768e-02],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 5.18319902e-01, 4.81680098e-01],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    ]))
    assert hmm._n_seqs == 3
    assert hmm._n_features == 3
    assert isinstance(hmm._model, pg.HiddenMarkovModel)

def test_load_invalid_path():
    """Load a GMMHMM from a directory"""
    with pytest.raises(IsADirectoryError) as e:
        GMMHMM.load('.')

def test_load_inexistent_path():
    """Load a GMMHMM from an inexistent path"""
    with pytest.raises(FileNotFoundError) as e:
        GMMHMM.load('test')

def test_load_invalid_format():
    """Load a GMMHMM from an illegally formatted file"""
    try:
        with open('test', 'w') as f:
            f.write('illegal')
        with pytest.raises(json.decoder.JSONDecodeError) as e:
            GMMHMM.load('test')
    finally:
        os.remove('test')

def test_load_invalid_json():
    """Load a GMMHMM from an invalid JSON file"""
    try:
        with open('test', 'w') as f:
            f.write("{}")
        with pytest.raises(KeyError) as e:
            GMMHMM.load('test')
    finally:
        os.remove('test')

def test_load_path():
    """Load a GMMHMM from a valid JSON file"""
    try:
        hmm = deepcopy(hmm_full)
        hmm.set_uniform_initial()
        hmm.set_uniform_transitions()
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            hmm.fit(X)
        hmm.save('test')
        hmm = GMMHMM.load('test')

        assert isinstance(hmm, GMMHMM)
        assert hmm._label == 'c1'
        assert hmm._n_states == 5
        assert hmm._n_components == 5
        assert isinstance(hmm._topology, _LeftRightTopology)
        assert hmm._covariance == 'full'
        assert_equal(hmm._initial, np.array([
            1.00000000e+000, 7.42563882e-211, 0.00000000e+000, 0.00000000e+000, 3.62089287e-116
        ]))
        assert_equal(hmm._transitions, np.array([
            [4.70308755e-01, 3.53127465e-01, 1.76563780e-01, 0.00000000e+00, 4.39410884e-63],
            [0.00000000e+00, 1.43953898e-01, 8.56046102e-01, 9.59514247e-23, 1.26148657e-53],
            [0.00000000e+00, 0.00000000e+00, 3.99999989e-01, 6.00000011e-01, 3.58675736e-22],
            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 5.00000064e-01, 4.99999936e-01],
            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
        ]))
        assert hmm._n_seqs == 3
        assert hmm._n_features == 3
        assert isinstance(hmm._model, pg.HiddenMarkovModel)
    finally:
        os.remove('test')