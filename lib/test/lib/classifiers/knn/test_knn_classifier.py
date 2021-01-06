import pytest, warnings, os, numpy as np, pickle
from copy import deepcopy
from multiprocessing import cpu_count
from sequentia.classifiers import KNNClassifier
from ....support import assert_equal, assert_all_equal, assert_not_equal

# Set seed for reproducible randomness
seed = 0
np.random.seed(seed)
rng = np.random.RandomState(seed)

classes = ['c0', 'c1', 'c2', 'c3', 'c4']

# Create some sample data
X = [rng.random((10 * i, 3)) for i in range(1, 7)]
y = ['c1', 'c1', 'c0', 'c1', 'c1', 'c0']
x = X[0]

clfs = {
    'k=1': KNNClassifier(k=1, classes=classes, random_state=rng),
    'k=2': KNNClassifier(k=2, classes=classes, random_state=rng),
    'k=3': KNNClassifier(k=3, classes=classes, random_state=rng),
    'weighted': KNNClassifier(k=3, classes=classes, weighting=(lambda x: np.exp(-x)), random_state=rng),
    'independent': KNNClassifier(k=1, classes=classes, independent=True, random_state=rng)
}

for _, clf in clfs.items():
    clf.fit(X, y)

# =================== #
# KNNClassifier.fit() #
# =================== #

def test_fit_sets_attributes():
    """Check that fitting sets the hidden attributes"""
    clf = clfs['k=1']
    clf.fit(X, y)
    assert_all_equal(clf._X, X)
    assert_equal(clf._y, clf._encoder.transform(y))
    assert clf._n_features == 3

# ======================= #
# KNNClassifier.predict() #
# ======================= #

def test_predict_without_fit():
    """Predict without fitting the model"""
    with pytest.raises(RuntimeError) as e:
        KNNClassifier(k=1, classes=classes).predict(x, verbose=False)
    assert str(e.value) == 'The classifier needs to be fitted before predictions are made'

def test_predict_single_k1_verbose(capsys):
    """Verbosely predict a single observation sequence (k=1)"""
    prediction = clfs['k=1'].predict(x, verbose=True)
    assert 'Calculating distances' in capsys.readouterr().err
    assert prediction == 'c1'

def test_predict_single_k1_no_verbose(capsys):
    """Silently predict a single observation sequence (k=1)"""
    prediction = clfs['k=1'].predict(x, verbose=False)
    assert 'Calculating distances' not in capsys.readouterr().err
    assert prediction == 'c1'

def test_predict_single_k2_verbose(capsys):
    """Verbosely predict a single observation sequence (k=2)"""
    prediction = clfs['k=2'].predict(x, verbose=True)
    assert 'Calculating distances' in capsys.readouterr().err
    assert prediction == 'c1'

def test_predict_single_k2_no_verbose(capsys):
    """Silently predict a single observation sequence (k=2)"""
    prediction = clfs['k=2'].predict(x, verbose=False)
    assert 'Calculating distances' not in capsys.readouterr().err
    assert prediction == 'c1'

def test_predict_single_k3_verbose(capsys):
    """Verbosely predict a single observation sequence (k=3)"""
    prediction = clfs['k=3'].predict(x, verbose=True)
    assert 'Calculating distances' in capsys.readouterr().err
    assert prediction == 'c1'

def test_predict_single_k3_no_verbose(capsys):
    """Silently predict a single observation sequence (k=3)"""
    prediction = clfs['k=3'].predict(x, verbose=False)
    assert 'Calculating distances' not in capsys.readouterr().err
    assert prediction == 'c1'

def test_predict_single_weighted_verbose(capsys):
    """Verbosely predict a single observation sequence (weighted)"""
    prediction = clfs['weighted'].predict(x, verbose=True)
    assert 'Calculating distances' in capsys.readouterr().err
    assert prediction == 'c1'

def test_predict_single_weighted_no_verbose(capsys):
    """Silently predict a single observation sequence (weighted)"""
    prediction = clfs['weighted'].predict(x, verbose=False)
    assert 'Calculating distances' not in capsys.readouterr().err
    assert prediction == 'c1'

def test_predict_single_independent_verbose(capsys):
    """Verbosely predict a single observation sequence with independent warping"""
    prediction = clfs['independent'].predict(x, verbose=True)
    assert 'Calculating distances' in capsys.readouterr().err
    assert prediction == 'c1'

def test_predict_single_k1_no_verbose(capsys):
    """Silently predict a single observation sequence with independent warping"""
    prediction = clfs['independent'].predict(x, verbose=False)
    assert 'Calculating distances' not in capsys.readouterr().err
    assert prediction == 'c1'

def test_predict_multiple_k1_verbose(capsys):
    """Verbosely predict multiple observation sequences (k=1)"""
    predictions = clfs['k=1'].predict(X, verbose=True)
    assert 'Classifying examples' in capsys.readouterr().err
    assert list(predictions) == ['c1', 'c1', 'c0', 'c1', 'c1', 'c0']

def test_predict_multiple_k1_no_verbose(capsys):
    """Silently predict multiple observation sequences (k=1)"""
    predictions = clfs['k=1'].predict(X, verbose=False)
    assert 'Classifying examples' not in capsys.readouterr().err
    assert list(predictions) == ['c1', 'c1', 'c0', 'c1', 'c1', 'c0']

def test_predict_multiple_k2_verbose(capsys):
    """Verbosely predict multiple observation sequences (k=2)"""
    predictions = clfs['k=2'].predict(X, verbose=True)
    assert 'Classifying examples' in capsys.readouterr().err
    assert len(predictions) == 6

def test_predict_multiple_k2_no_verbose(capsys):
    """Silently predict multiple observation sequences (k=2)"""
    predictions = clfs['k=2'].predict(X, verbose=False)
    assert 'Classifying examples' not in capsys.readouterr().err
    assert len(predictions) == 6

def test_predict_multiple_k3_verbose(capsys):
    """Verbosely predict multiple observation sequences (k=3)"""
    predictions = clfs['k=3'].predict(X, verbose=True)
    assert 'Classifying examples' in capsys.readouterr().err
    assert list(predictions) == ['c1', 'c1', 'c1', 'c1', 'c0', 'c1']

def test_predict_multiple_k3_no_verbose(capsys):
    """Silently predict multiple observation sequences (k=3)"""
    predictions = clfs['k=3'].predict(X, verbose=False)
    assert 'Classifying examples' not in capsys.readouterr().err
    assert list(predictions) == ['c1', 'c1', 'c1', 'c1', 'c0', 'c1']

def test_predict_multiple_weighted_verbose(capsys):
    """Verbosely predict multiple observation sequences (weighted)"""
    predictions = clfs['weighted'].predict(X, verbose=True)
    assert 'Classifying examples' in capsys.readouterr().err
    assert list(predictions) == ['c1', 'c1', 'c0', 'c1', 'c0', 'c1']

def test_predict_multiple_weighted_no_verbose(capsys):
    """Silently predict multiple observation sequences (weighted)"""
    predictions = clfs['weighted'].predict(X, verbose=False)
    assert 'Classifying examples' not in capsys.readouterr().err
    assert list(predictions) == ['c1', 'c1', 'c0', 'c1', 'c0', 'c1']

def test_predict_multiple_independent_verbose(capsys):
    """Verbosely predict multiple observation sequences with independent warping"""
    predictions = clfs['independent'].predict(X, verbose=True)
    assert 'Classifying examples' in capsys.readouterr().err
    assert list(predictions) == ['c1', 'c1', 'c0', 'c1', 'c1', 'c0']

def test_predict_multiple_independent_no_verbose(capsys):
    """Silently predict multiple observation sequences with independent warping"""
    predictions = clfs['independent'].predict(X, verbose=False)
    assert 'Classifying examples' not in capsys.readouterr().err
    assert list(predictions) == ['c1', 'c1', 'c0', 'c1', 'c1', 'c0']

def test_predict_single():
    """Predict a single observation sequence and don't return the original labels"""
    prediction = clfs['k=3'].predict(x, verbose=False, original_labels=False)
    assert prediction == 1

def test_predict_single_original_labels():
    """Predict a single observation sequence and return the original labels"""
    prediction = clfs['k=3'].predict(x, verbose=False, original_labels=True)
    assert prediction == 'c1'

def test_predict_multiple():
    """Predict multiple observation sequences and don't return the original labels"""
    predictions = clfs['k=3'].predict(X, verbose=False, original_labels=False)
    assert list(predictions) == [1, 1, 1, 1, 0, 1]

def test_predict_multiple_original_labels():
    """Predict multiple observation sequences and return the original labels"""
    predictions = clfs['k=3'].predict(X, verbose=False, original_labels=True)
    assert list(predictions) == ['c1', 'c1', 'c1', 'c1', 'c0', 'c1']

# ======================== #
# KNNClassifier.evaluate() #
# ======================== #

def test_evaluate():
    """Evaluate performance on some observation sequences and labels"""
    acc, cm = clfs['k=3'].evaluate(X, y)
    assert acc == 0.5
    assert_equal(cm, np.array([
        [0, 2, 0, 0, 0],
        [1, 3, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ]))

# ==================== #
# KNNClassifier.save() #
# ==================== #

def test_save_unfitted():
    """Save an unfitted KNNClassifier object."""
    try:
        with pytest.raises(RuntimeError) as e:
            KNNClassifier(k=1, classes=classes).save('test.pkl')
        assert str(e.value) == 'The classifier needs to be fitted before it can be saved'
    finally:
        if os.path.exists('test.pkl'):
            os.remove('test.pkl')

def test_save_fitted():
    """Save a fitted KNNClassifier object."""
    try:
        clfs['weighted'].save('test.pkl')
        assert os.path.isfile('test.pkl')
    finally:
        os.remove('test.pkl')

# ==================== #
# KNNClassifier.load() #
# ==================== #

def test_load_invalid_format():
    """Load a KNNClassifier from an illegally formatted file"""
    try:
        with open('test', 'w') as f:
            f.write('illegal')
        with pytest.raises(pickle.UnpicklingError) as e:
            KNNClassifier.load('test')
    finally:
        os.remove('test')

def test_load_invalid_object_type():
    """Load a KNNClassifier from a pickle file with invalid object type"""
    try:
        with open('test.pkl', 'wb') as file:
            pickle.dump(0, file)
        with pytest.raises(TypeError) as e:
            KNNClassifier.load('test.pkl')
        assert str(e.value) == 'Expected deserialized object to be a dictionary - make sure the object was serialized with the save() function'
    finally:
        os.remove('test.pkl')

def test_load_missing_keys():
    """Load a KNNClassifier from a pickled dict with invalid keys"""
    try:
        with open('test.pkl', 'wb') as file:
            pickle.dump({'test': 0}, file)
        with pytest.raises(ValueError) as e:
            KNNClassifier.load('test.pkl')
        assert str(e.value) == 'Missing keys in deserialized object dictionary – make sure the object was serialized with the save() function'
    finally:
        os.remove('test.pkl')

def test_load_valid_no_weighting():
    """Load a serialized KNNClassifier with the default weighting function"""
    try:
        clfs['k=3'].save('test.pkl')
        clf = KNNClassifier.load('test.pkl')
        # Check that all fields are still the same
        assert isinstance(clf, KNNClassifier)
        assert clf._k == 3
        assert list(clf._encoder.classes_) == classes
        assert clf._window == 1.
        assert clf._use_c == False
        assert clf._independent == False
        assert deepcopy(clf._random_state).normal() == deepcopy(rng).normal()
        assert_all_equal(clf._X, X)
        assert_equal(clf._y, clf._encoder.transform(y))
        assert clf._n_features == 3
        # Check that weighting functions are the same for x=0 to x=1000
        xs = np.arange(1000, step=0.1)
        weighting = lambda x: np.ones(x.size)
        assert_equal(clf._weighting(xs), weighting(xs))
    finally:
        os.remove('test.pkl')

def test_load_valid_weighting():
    """Load a serialized KNNClassifier with a custom weighting function"""
    try:
        clfs['weighted'].save('test.pkl')
        clf = KNNClassifier.load('test.pkl')
        # Check that all fields are still the same
        assert isinstance(clf, KNNClassifier)
        assert clf._k == 3
        assert list(clf._encoder.classes_) == classes
        assert clf._window == 1.
        assert clf._use_c == False
        assert clf._independent == False
        assert deepcopy(clf._random_state).normal() == deepcopy(rng).normal()
        assert_all_equal(clf._X, X)
        assert_equal(clf._y, clf._encoder.transform(y))
        assert clf._n_features == 3
        # Check that weighting functions are the same for x=0 to x=1000
        xs = np.arange(1000, step=0.1)
        weighting = lambda x: np.exp(-x)
        assert_equal(clf._weighting(xs), weighting(xs))
    finally:
        os.remove('test.pkl')