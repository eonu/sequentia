import scipy.fftpack
import numpy as np
from typing import Union, List

def normalize(X: Union[np.ndarray, List[np.ndarray]]) -> Union[np.ndarray, List[np.ndarray]]:
    """Normalizes an observation sequence (or multiple sequences) by centering observations around the mean.

    Parameters:
        X {numpy.ndarray, list(numpy.ndarray)} - An individual observation sequence or
            a list of multiple observation sequences.

    Returns {numpy.ndarray, list(numpy.ndarray)}:
        The normalized input observation sequence(s).
    """
    if isinstance(X, list):
        if not all(isinstance(sequence, np.ndarray) for sequence in X):
            raise TypeError('Each observation sequence must be a numpy.ndarray')
        if not all(sequence.ndim == 2 for sequence in X):
            raise ValueError('Each observation sequence must be two-dimensional')
        if not all(sequence.shape[1] == X[0].shape[1] for sequence in X):
            raise ValueError('Each observation sequence must have the same dimensionality')
    elif isinstance(X, np.ndarray):
        if not X.ndim == 2:
            raise ValueError('Sequence of observations must be two-dimensional')
    else:
        raise TypeError('Expected a single observation sequence (numpy.ndarray) or list of observation sequences')
    return _normalize(X)

def _normalize(X: Union[np.ndarray, List[np.ndarray]]) -> Union[np.ndarray, List[np.ndarray]]:
    def transform(x):
        return x - x.mean(axis=0)

    if isinstance(X, list):
        return [transform(x) for x in X]
    elif isinstance(X, np.ndarray):
        return transform(X)

def downsample(X: Union[np.ndarray, List[np.ndarray]], n: int, method='decimate') -> Union[np.ndarray, List[np.ndarray]]:
    """Downsamples an observation sequence (or multiple sequences) by:
        - Decimating the next n-1 observations
        - Averaging the current observation with the next n-1 observations

    Parameters:
        X {numpy.ndarray, list(numpy.ndarray)} - An individual observation sequence or
            a list of multiple observation sequences.
        n {int} - Downsample factor.
            NOTE: This downsamples the current observation by either decimating the next n-1
                observations or computing an average with them.
        method {str} - The downsamplimg method, either 'decimate' or 'average'.

    Returns {numpy.ndarray, list(numpy.ndarray)}:
        The downsampled input observation sequence(s).
    """
    if isinstance(X, list):
        if not all(isinstance(sequence, np.ndarray) for sequence in X):
            raise TypeError('Each observation sequence must be a numpy.ndarray')
        if not all(sequence.ndim == 2 for sequence in X):
            raise ValueError('Each observation sequence must be two-dimensional')
        if not all(sequence.shape[1] == X[0].shape[1] for sequence in X):
            raise ValueError('Each observation sequence must have the same dimensionality')
    elif isinstance(X, np.ndarray):
        if not X.ndim == 2:
            raise ValueError('Sequence of observations must be two-dimensional')
    else:
        raise TypeError('Expected a single observation sequence (numpy.ndarray) or list of observation sequences')

    if not isinstance(n, int):
        raise TypeError('Expected downsample factor to be an integer')
    if not n > 1:
        raise ValueError('Expected downsample factor to be greater than one')

    if method not in ['decimate', 'average']:
        raise ValueError("Expected downsample method to be one of 'decimate' or 'average'")

    return _downsample(X, n, method)

def _downsample(X: Union[np.ndarray, List[np.ndarray]], n: int, method: str) -> Union[np.ndarray, List[np.ndarray]]:
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

def fft(X: Union[np.ndarray, List[np.ndarray]]) -> Union[np.ndarray, List[np.ndarray]]:
    """Applies a Discrete Fourier Transform to the input observation sequence(s).

    Returns {numpy.ndarray, list(numpy.ndarray)}:
        The transformed input observation sequence(s).
    """
    if isinstance(X, list):
        if not all(isinstance(sequence, np.ndarray) for sequence in X):
            raise TypeError('Each observation sequence must be a numpy.ndarray')
        if not all(sequence.ndim == 2 for sequence in X):
            raise ValueError('Each observation sequence must be two-dimensional')
        if not all(sequence.shape[1] == X[0].shape[1] for sequence in X):
            raise ValueError('Each observation sequence must have the same dimensionality')
    elif isinstance(X, np.ndarray):
        if not X.ndim == 2:
            raise ValueError('Sequence of observations must be two-dimensional')
    else:
        raise TypeError('Expected a single observation sequence (numpy.ndarray) or list of observation sequences')

    return _fft(X)

def _fft(X: Union[np.ndarray, List[np.ndarray]]) -> Union[np.ndarray, List[np.ndarray]]:
    def transform(x):
        return scipy.fftpack.rfft(x, axis=0)

    if isinstance(X, list):
        return [transform(x) for x in X]
    elif isinstance(X, np.ndarray):
        return transform(X)