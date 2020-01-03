from numpy.testing import assert_allclose

def assert_arrays_equal(a, b):
    assert_allclose(a, b, rtol=1e-5)

def assert_all_arrays_equal(A, B):
    for a, b in zip(A, B):
        assert_arrays_equal(a, b)