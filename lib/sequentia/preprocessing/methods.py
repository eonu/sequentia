import scipy.fftpack
import numpy as np
from ..internals import Validator

def normalize(X):
    """Normalizes an observation sequence (or multiple sequences) by centering observations around the mean.

    Parameters
    ----------
    X: numpy.ndarray or List[numpy.ndarray]
        An individual observation sequence or a list of multiple observation sequences.

    Returns
    -------
    normalized: numpy.ndarray or List[numpy.ndarray]
        The normalized input observation sequence(s).
    """
    val = Validator()
    val.observation_sequences(X, allow_single=True)
    return _normalize(X)

def _normalize(X):
    def transform(x):
        return x - x.mean(axis=0)

    if isinstance(X, list):
        return [transform(x) for x in X]
    elif isinstance(X, np.ndarray):
        return transform(X)

def downsample(X, n, method='decimate'):
    """Downsamples an observation sequence (or multiple sequences) by either:
        - Decimating the next :math:`n-1` observations
        - Averaging the current observation with the next :math:`n-1` observations

    Parameters
    ----------
    X: numpy.ndarray or List[numpy.ndarray]
        An individual observation sequence or a list of multiple observation sequences.

    n: int
        Downsample factor.

    method: {'decimate', 'average'}
        The downsampling method.

    Returns
    -------
    downsampled: numpy.ndarray or List[numpy.ndarray]
        The downsampled input observation sequence(s).
    """
    val = Validator()
    val.observation_sequences(X, allow_single=True)
    val.restricted_integer(n, lambda x: x > 1, desc='downsample factor', expected='greater than one')
    val.one_of(method, ['decimate', 'average'], desc='downsampling method')
    return _downsample(X, n, method)

def _downsample(X, n, method):
    def transform(x):
        N, D = x.shape
        if method == 'decimate':
            return np.delete(x, [i for i in range(N) if i % n != 0], 0)
        elif method == 'average':
            pad = (n - (len(x) % n)) % n
            padded = np.vstack((x, np.tile(x[-1, :], (pad, 1))))
            return padded.T.reshape(-1, n).mean(1).reshape(D, -1).T

    if isinstance(X, list):
        return [transform(x) for x in X]
    elif isinstance(X, np.ndarray):
        return transform(X)

def fft(X):
    """Applies a Discrete Fourier Transform to the input observation sequence(s).

    Parameters
    ----------
    X: numpy.ndarray or List[numpy.ndarray]
        An individual observation sequence or a list of multiple observation sequences.

    Returns
    -------
    transformed: numpy.ndarray or List[numpy.ndarray]
        The transformed input observation sequence(s).
    """
    val = Validator()
    val.observation_sequences(X, allow_single=True)
    return _fft(X)

def _fft(X):
    def transform(x):
        return scipy.fftpack.rfft(x, axis=0)

    if isinstance(X, list):
        return [transform(x) for x in X]
    elif isinstance(X, np.ndarray):
        return transform(X)