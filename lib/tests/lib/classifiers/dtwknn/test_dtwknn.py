import pytest
import warnings
import numpy as np
from copy import deepcopy
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    from sequentia.classifiers import DTWKNN
from ....support import assert_equal, assert_not_equal

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
    DTWKNN(k=1, radius=1),
    DTWKNN(k=2, radius=5),
    DTWKNN(k=3, radius=10)
]

for clf in clfs:
    clf.fit(X, y)

# ============ #
# DTWKNN.fit() #
# ============ #

def test_fit_sets_attributes():
    """Check that fitting sets the hidden attributes"""
    clf = clfs[0]
    clf.fit(X, y)
    assert clf._X == X
    assert clf._y == y

# ================ #
# DTWKNN.predict() #
# ================ #

def test_predict_without_fit():
    """Predict without fitting the model"""
    with pytest.raises(RuntimeError) as e:
        DTWKNN(k=1, radius=1).predict(x, verbose=False)
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

# ================= #
# DTWKNN.evaluate() #
# ================= #

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