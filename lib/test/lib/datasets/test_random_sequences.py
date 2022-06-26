import pytest, numpy as np
from itertools import product
from sequentia.datasets.random_sequences import load_random_sequences, _sample_prior, _K, _sample_from_range
from ...support import assert_equal

random_state = np.random.RandomState(0)

K = lambda x1, x2: _K(x1, x2, variance=1, lengthscale=1)

# load_random_sequences

@pytest.mark.parametrize(
    'n_sequences,n_features,n_classes,length_range',
    list(product(
        [1, 10, 25, 30],
        [1, 2, 3],
        [1, 2, 5],
        [(5, 15), 15]
    ))
)
def test_load_random_sequences(n_sequences, n_features, n_classes, length_range):
    if n_classes > n_sequences:

        with pytest.raises(Exception) as e:
            load_random_sequences(
                n_sequences, n_features, n_classes, length_range,
                random_state=random_state
            )

        assert str(e.value) == 'Expected number of classes to be no more than n_sequences'

    else:

        dataset = load_random_sequences(
            n_sequences, n_features, n_classes, length_range,
            random_state=random_state
        )

        # Checking dataset size
        assert len(dataset) == n_sequences

        # Checking classes
        assert isinstance(dataset.y, np.ndarray)
        assert dataset.y.shape == (n_sequences,)
        assert set(dataset.y) == set(range(n_classes))

        # Checking sequence lengths and number of features
        if isinstance(length_range, tuple):
            a, b = length_range
            assert all((a <= len(x) <= b) and x.shape[1] == n_features for x in dataset.X)
        else:
            X = np.array(dataset.X)
            assert X.shape == (n_sequences, length_range, n_features)

# _sample_prior

@pytest.mark.parametrize(
    'length,n_features',
    list(product([5, 10, 15], [1, 2, 5]))
)
def test_sample_prior(length, n_features):
    y = _sample_prior(K, length, n_features, random_state)
    assert y.shape == (length, n_features)

# _K

def test_K():
    X = np.expand_dims(np.arange(5), axis=1)
    cov = K(X, X)
    assert_equal(cov, np.array([
        [1.00000000e+00, 6.06530660e-01, 1.35335283e-01, 1.11089965e-02, 3.35462628e-04],
        [6.06530660e-01, 1.00000000e+00, 6.06530660e-01, 1.35335283e-01, 1.11089965e-02],
        [1.35335283e-01, 6.06530660e-01, 1.00000000e+00, 6.06530660e-01, 1.35335283e-01],
        [1.11089965e-02, 1.35335283e-01, 6.06530660e-01, 1.00000000e+00, 6.06530660e-01],
        [3.35462628e-04, 1.11089965e-02, 1.35335283e-01, 6.06530660e-01, 1.00000000e+00]
    ]))