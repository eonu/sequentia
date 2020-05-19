import pytest, warnings, os, h5py, numpy as np
from copy import deepcopy
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    from sequentia.classifiers import KNNClassifier
from ....support import assert_equal, assert_all_equal, assert_not_equal

# Set seed for reproducible randomness
seed = 0
np.random.seed(seed)
rng = np.random.RandomState(seed)

labels = ['c0', 'c1', 'c2', 'c3', 'c4']

# Create some sample data
X = [rng.random((10 * i, 3)) for i in range(1, 7)]
y = ['c1', 'c1', 'c0', 'c1', 'c1', 'c0']
x = X[0]

clfs = [
    KNNClassifier(k=1, radius=1),
    KNNClassifier(k=2, radius=5),
    KNNClassifier(k=3, radius=10)
]

for clf in clfs:
    clf.fit(X, y)

# =================== #
# KNNClassifier.fit() #
# =================== #

def test_fit_sets_attributes():
    """Check that fitting sets the hidden attributes"""
    clf = clfs[0]
    clf.fit(X, y)
    assert clf._X == X
    assert clf._y == y

# ======================= #
# KNNClassifier.predict() #
# ======================= #

def test_predict_without_fit():
    """Predict without fitting the model"""
    with pytest.raises(RuntimeError) as e:
        KNNClassifier(k=1, radius=1).predict(x, verbose=False)
    assert str(e.value) == 'The classifier needs to be fitted before predictions are made'

def test_predict_single_k1_r1_verbose(capsys):
    """Verbosely predict a single observation sequence (k=1, r=1)"""
    prediction = clfs[0].predict(x, verbose=True)
    assert 'Calculating distances' in capsys.readouterr().err
    assert prediction == 'c1'

def test_predict_single_k1_r1_no_verbose(capsys):
    """Silently predict a single observation sequence (k=1, r=1)"""
    prediction = clfs[0].predict(x, verbose=False)
    assert 'Calculating distances' not in capsys.readouterr().err
    assert prediction == 'c1'

def test_predict_single_k2_r5_verbose(capsys):
    """Verbosely predict a single observation sequence (k=2, r=5)"""
    prediction = clfs[1].predict(x, verbose=True)
    assert 'Calculating distances' in capsys.readouterr().err
    assert prediction == 'c1'

def test_predict_single_k2_r5_no_verbose(capsys):
    """Silently predict a single observation sequence (k=2, r=5)"""
    prediction = clfs[1].predict(x, verbose=False)
    assert 'Calculating distances' not in capsys.readouterr().err
    assert prediction == 'c1'

def test_predict_single_k3_r10_verbose(capsys):
    """Verbosely predict a single observation sequence (k=3, r=10)"""
    prediction = clfs[2].predict(x, verbose=True)
    assert 'Calculating distances' in capsys.readouterr().err
    assert prediction == 'c1'

def test_predict_single_k3_r10_no_verbose(capsys):
    """Silently predict a single observation sequence (k=3, r=10)"""
    prediction = clfs[2].predict(x, verbose=False)
    assert 'Calculating distances' not in capsys.readouterr().err
    assert prediction == 'c1'

def test_predict_multiple_k1_r1_verbose(capsys):
    """Verbosely predict multiple observation sequences (k=1, r=1)"""
    predictions = clfs[0].predict(X, verbose=True)
    assert 'Classifying examples' in capsys.readouterr().err
    assert predictions == ['c1', 'c1', 'c0', 'c1', 'c1', 'c0']

def test_predict_multiple_k1_r1_no_verbose(capsys):
    """Silently predict multiple observation sequences (k=1, r=1)"""
    predictions = clfs[0].predict(X, verbose=False)
    assert 'Classifying examples' not in capsys.readouterr().err
    assert predictions == ['c1', 'c1', 'c0', 'c1', 'c1', 'c0']

def test_predict_multiple_k2_r5_verbose(capsys):
    """Verbosely predict multiple observation sequences (k=2, r=5)"""
    predictions = clfs[1].predict(X, verbose=True)
    assert 'Classifying examples' in capsys.readouterr().err
    assert len(predictions) == 6

def test_predict_multiple_k2_r5_no_verbose(capsys):
    """Silently predict multiple observation sequences (k=2, r=5)"""
    predictions = clfs[1].predict(X, verbose=False)
    assert 'Classifying examples' not in capsys.readouterr().err
    assert len(predictions) == 6

def test_predict_multiple_k3_r10_verbose(capsys):
    """Verbosely predict multiple observation sequences (k=3, r=10)"""
    predictions = clfs[2].predict(X, verbose=True)
    assert 'Classifying examples' in capsys.readouterr().err
    assert len(predictions) == 6

def test_predict_multiple_k3_r10_no_verbose(capsys):
    """Silently predict multiple observation sequences (k=3, r=10)"""
    predictions = clfs[2].predict(X, verbose=False)
    assert 'Classifying examples' not in capsys.readouterr().err
    assert len(predictions) == 6

# ======================== #
# KNNClassifier.evaluate() #
# ======================== #

def test_evaluate_with_labels_k1_r1_verbose(capsys):
    """Verbosely evaluate observation sequences with labels (k=1, r=1)"""
    acc, cm = clfs[0].evaluate(X, y, labels=labels, verbose=True)
    assert 'Classifying examples' in capsys.readouterr().err
    assert isinstance(acc, float)
    assert isinstance(cm, np.ndarray)
    assert cm.shape == (5, 5)

def test_evaluate_with_labels_k1_r1_no_verbose(capsys):
    """Silently evaluate observation sequences with labels (k=1, r=1)"""
    acc, cm = clfs[0].evaluate(X, y, labels=labels, verbose=False)
    assert 'Classifying examples' not in capsys.readouterr().err
    assert isinstance(acc, float)
    assert isinstance(cm, np.ndarray)
    assert cm.shape == (5, 5)

def test_evaluate_with_no_labels_k1_r1_verbose(capsys):
    """Verbosely evaluate observation sequences without labels (k=1, r=1)"""
    acc, cm = clfs[0].evaluate(X, y, labels=None, verbose=True)
    assert 'Classifying examples' in capsys.readouterr().err
    assert isinstance(acc, float)
    assert isinstance(cm, np.ndarray)

def test_evaluate_with_no_labels_k1_r1_no_verbose(capsys):
    """Silently evaluate observation sequences without labels (k=1, r=1)"""
    acc, cm = clfs[0].evaluate(X, y, labels=None, verbose=False)
    assert 'Classifying examples' not in capsys.readouterr().err
    assert isinstance(acc, float)
    assert isinstance(cm, np.ndarray)

def test_evaluate_with_labels_k2_r5_verbose(capsys):
    """Verbosely evaluate observation sequences with labels (k=2, r=5)"""
    acc, cm = clfs[1].evaluate(X, y, labels=labels, verbose=True)
    assert 'Classifying examples' in capsys.readouterr().err
    assert isinstance(acc, float)
    assert isinstance(cm, np.ndarray)
    assert cm.shape == (5, 5)

def test_evaluate_with_labels_k2_r5_no_verbose(capsys):
    """Silently evaluate observation sequences with labels (k=2, r=5)"""
    acc, cm = clfs[1].evaluate(X, y, labels=labels, verbose=False)
    assert 'Classifying examples' not in capsys.readouterr().err
    assert isinstance(acc, float)
    assert isinstance(cm, np.ndarray)
    assert cm.shape == (5, 5)

def test_evaluate_with_no_labels_k2_r5_verbose(capsys):
    """Verbosely evaluate observation sequences without labels (k=2, r=5)"""
    acc, cm = clfs[1].evaluate(X, y, labels=None, verbose=True)
    assert 'Classifying examples' in capsys.readouterr().err
    assert isinstance(acc, float)
    assert isinstance(cm, np.ndarray)

def test_evaluate_with_no_labels_k2_r5_no_verbose(capsys):
    """Silently evaluate observation sequences without labels (k=2, r=5)"""
    acc, cm = clfs[1].evaluate(X, y, labels=None, verbose=False)
    assert 'Classifying examples' not in capsys.readouterr().err
    assert isinstance(acc, float)
    assert isinstance(cm, np.ndarray)

def test_evaluate_with_labels_k3_r10_verbose(capsys):
    """Verbosely evaluate observation sequences with labels (k=3, r=10)"""
    acc, cm = clfs[2].evaluate(X, y, labels=labels, verbose=True)
    assert 'Classifying examples' in capsys.readouterr().err
    assert isinstance(acc, float)
    assert isinstance(cm, np.ndarray)
    assert cm.shape == (5, 5)

def test_evaluate_with_labels_k3_r10_no_verbose(capsys):
    """Silently evaluate observation sequences with labels (k=3, r=10)"""
    acc, cm = clfs[2].evaluate(X, y, labels=labels, verbose=False)
    assert 'Classifying examples' not in capsys.readouterr().err
    assert isinstance(acc, float)
    assert isinstance(cm, np.ndarray)
    assert cm.shape == (5, 5)

def test_evaluate_with_no_labels_k3_r10_verbose(capsys):
    """Verbosely evaluate observation sequences without labels (k=3, r=10)"""
    acc, cm = clfs[2].evaluate(X, y, labels=None, verbose=True)
    assert 'Classifying examples' in capsys.readouterr().err
    assert isinstance(acc, float)
    assert isinstance(cm, np.ndarray)

def test_evaluate_with_no_labels_k3_r10_no_verbose(capsys):
    """Silently evaluate observation sequences without labels (k=3, r=10)"""
    acc, cm = clfs[2].evaluate(X, y, labels=None, verbose=False)
    assert 'Classifying examples' not in capsys.readouterr().err
    assert isinstance(acc, float)
    assert isinstance(cm, np.ndarray)

# ==================== #
# KNNClassifier.save() #
# ==================== #

def test_save_directory():
    """Save a KNNClassifier classifier into a directory"""
    with pytest.raises(OSError) as e:
        clfs[2].save('.')

def test_save_no_extension():
    """Save a KNNClassifier classifier into a file without an extension"""
    try:
        clfs[2].save('test')
        assert os.path.isfile('test')
    finally:
        os.remove('test')

def test_save_with_extension():
    """Save a KNNClassifier classifier into a file with a .h5 extension"""
    try:
        clfs[2].save('test.h5')
        assert os.path.isfile('test.h5')
    finally:
        os.remove('test.h5')

# ==================== #
# KNNClassifier.load() #
# ==================== #

def test_load_invalid_path():
    """Load a KNNClassifier classifier from a directory"""
    with pytest.raises(OSError) as e:
        KNNClassifier.load('.')

def test_load_inexistent_path():
    """Load a KNNClassifier classifier from an inexistent path"""
    with pytest.raises(OSError) as e:
        KNNClassifier.load('test')

def test_load_invalid_format():
    """Load a KNNClassifier classifier from an illegally formatted file"""
    try:
        with open('test', 'w') as f:
            f.write('illegal')
        with pytest.raises(OSError) as e:
            KNNClassifier.load('test')
    finally:
        os.remove('test')

def test_load_path():
    """Load a KNNClassifier classifier from a valid HDF5 file"""
    try:
        clfs[2].save('test')
        clf = KNNClassifier.load('test')

        assert isinstance(clf, KNNClassifier)
        assert clf._k == 3
        assert clf._radius == 10
        assert isinstance(clf._X, list)
        assert len(clf._X) == len(X)
        assert isinstance(clf._X[0], np.ndarray)
        assert_all_equal(clf._X, X)
        assert isinstance(clf._y, list)
        assert len(clf._y) == len(y)
        assert isinstance(clf._y[0], str)
        assert all(y1 == y2 for y1, y2 in zip(clf._y, y))
    finally:
        os.remove('test')