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
    before = hmm.initial, hmm.transitions
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
    assert_not_equal(d['model']['initial'], before[0])
    assert_not_equal(d['model']['transitions'], before[1])
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
    try:
        with pytest.raises(IsADirectoryError) as e:
            hmm.save('.')
        assert str(e.value) == "[Errno 21] Is a directory: '.'"
    except ValueError:
        pass

def test_save_no_extension():
    """Save a GMMHMM into a file without an extension"""
    try:
        hmm = deepcopy(hmm_diag)
        hmm.set_uniform_initial()
        hmm.set_uniform_transitions()
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            hmm.fit(X)
        try:
            hmm.save('test')
            assert os.path.isfile('test')
        except ValueError:
            pass
    finally:
        try:
            os.remove('test')
        except OSError:
            pass

def test_save_with_extension():
    """Save a GMMHMM into a file with a .json extension"""
    try:
        hmm = deepcopy(hmm_diag)
        hmm.set_uniform_initial()
        hmm.set_uniform_transitions()
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            hmm.fit(X)
        try:
            hmm.save('test.json')
            assert os.path.isfile('test.json')
        except ValueError:
            pass
    finally:
        try:
            os.remove('test.json')
        except OSError:
            pass

# ============= #
# GMMHMM.load() #
# ============= #

def test_load_invalid_dict():
    """Load a GMMHMM from an invalid dict"""
    with pytest.raises(KeyError) as e:
        GMMHMM.load({})

def test_load_dict():
    """Load a GMMHMM from a valid dict"""
    hmm = deepcopy(hmm_diag)
    hmm.set_uniform_initial()
    hmm.set_uniform_transitions()
    before = hmm.initial, hmm.transitions
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
    assert_not_equal(hmm._initial, before[0])
    assert_not_equal(hmm._transitions, before[1])
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
        hmm = deepcopy(hmm_diag)
        hmm.set_uniform_initial()
        hmm.set_uniform_transitions()
        before = hmm.initial, hmm.transitions
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
        assert hmm._covariance == 'diagonal'
        assert_not_equal(hmm._initial, before[0])
        assert_not_equal(hmm._transitions, before[1])
        assert hmm._n_seqs == 3
        assert hmm._n_features == 3
        assert isinstance(hmm._model, pg.HiddenMarkovModel)
    finally:
        os.remove('test')