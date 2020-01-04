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

# Create some sample data
X = [rng.random((10 * i, 3)) for i in range(1, 7)]
y = ['c1', 'c1', 'c0', 'c1', 'c1', 'c0']
x = X[0]

clfs = [
    DTWKNN(k=1, radius=1),
    DTWKNN(k=2, radius=5),
    DTWKNN(k=3, radius=10)
]

# ============ #
# DTWKNN.fit() #
# ============ #

def test_fit():
    """"""
    clf = clfs[0]
    clf.fit(X, y)
    assert clf._X == X
    assert clf._y == y

def test_predict_without_fit():
    """"""
    pass

def test_predict_single_k1_r1_verbose(capsys):
    """Verbosely predict a single observation sequence (k=1, r=1)"""
    clf = clfs[0]
    clf.fit(X, y)
    prediction = clf.predict(x, verbose=True)
    assert 'Calculating distances' in capsys.readouterr().err
    assert prediction == 'c1'

def test_predict_single_k1_r1_no_verbose(capsys):
    """Silently predict a single observation sequence (k=1, r=1)"""
    clf = clfs[0]
    clf.fit(X, y)
    prediction = clf.predict(x, verbose=False)
    assert 'Calculating distances' not in capsys.readouterr().err
    assert prediction == 'c1'

def test_predict_single_k2_r5_verbose(capsys):
    """Verbosely predict a single observation sequence (k=2, r=5)"""
    clf = clfs[1]
    clf.fit(X, y)
    prediction = clf.predict(x, verbose=True)
    assert 'Calculating distances' in capsys.readouterr().err
    assert prediction == 'c1'

def test_predict_single_k2_r5_no_verbose(capsys):
    """Silently predict a single observation sequence (k=2, r=5)"""
    clf = clfs[1]
    clf.fit(X, y)
    prediction = clf.predict(x, verbose=False)
    assert 'Calculating distances' not in capsys.readouterr().err
    assert prediction == 'c1'

def test_predict_single_k3_r10_verbose(capsys):
    """Verbosely predict a single observation sequence (k=3, r=10)"""
    clf = clfs[2]
    clf.fit(X, y)
    prediction = clf.predict(x, verbose=True)
    assert 'Calculating distances' in capsys.readouterr().err
    assert prediction == 'c1'

def test_predict_single_k3_r10_no_verbose(capsys):
    """Silently predict a single observation sequence (k=3, r=10)"""
    clf = clfs[2]
    clf.fit(X, y)
    prediction = clf.predict(x, verbose=False)
    assert 'Calculating distances' not in capsys.readouterr().err
    assert prediction == 'c1'