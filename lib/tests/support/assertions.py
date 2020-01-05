import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal

def assert_equal(a, b):
    assert_allclose(a, b, rtol=1e-5)

def assert_not_equal(a, b):
    assert not np.allclose(a, b, rtol=1e-5)

def assert_all_equal(A, B):
    for a, b in zip(A, B):
        assert_equal(a, b)

def assert_distribution(a):
    if a.ndim == 1:
        assert_almost_equal(a.sum(), 1., decimal=5)
    elif a.ndim == 2:
        assert_almost_equal(a.sum(axis=1), np.ones(len(a)))