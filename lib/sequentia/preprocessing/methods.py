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

    if isinstance(X, np.ndarray):
        val.restricted_integer(n, lambda x: x <= len(X),
            desc='downsample factor', expected='no greater than the number of frames')
    else:
        val.restricted_integer(n, lambda x: x <= min(len(x) for x in X),
            desc='downsample factor', expected='no greater than the number of frames in the shortest sequence')

    return _downsample(X, n, method)

def _downsample(X, n, method):
    def transform(x):
        N, D = x.shape
        if method == 'decimate':
            return np.delete(x, [i for i in range(N) if i % n != 0], 0)
        elif method == 'average':
            r = len(x) % n
            xn, xr = (x, None) if r == 0 else (x[:-r], x[-r:])
            dxn = xn.T.reshape(-1, n).mean(axis=1).reshape(D, -1).T
            return dxn if xr is None else np.vstack((dxn, xr.mean(axis=0)))

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

def filtrate(X, n, method='median'):
    """Applies a median or mean filter to the input observation sequence(s).

    Parameters
    ----------
    X: numpy.ndarray or List[numpy.ndarray]
        An individual observation sequence or a list of multiple observation sequences.

    n: int
        Window size.

    method: {'median', 'mean'}
        The filtering method.

    Returns
    -------
    filtered: numpy.ndarray or List[numpy.ndarray]
        The filtered input observation sequence(s).
    """
    val = Validator()
    val.observation_sequences(X, allow_single=True)
    val.restricted_integer(n, lambda x: x > 1, desc='window size', expected='greater than one')
    val.one_of(method, ['median', 'mean'], desc='filtering method')

    if isinstance(X, np.ndarray):
        val.restricted_integer(n, lambda x: x <= len(X),
            desc='window size', expected='no greater than the number of frames')
    else:
        val.restricted_integer(n, lambda x: x <= min(len(x) for x in X),
            desc='window size', expected='no greater than the number of frames in the shortest sequence')

    return _filtrate(X, n, method)

def _filtrate(X, n, method):
    def transform(x):
        measure = np.median if method == 'median' else np.mean
        filtered = []
        right = n // 2
        left = (n - 1) - right
        for i in range(len(x)):
            l, m, r = x[((i - left) * (left < i)):i], x[i], x[(i + 1):(i + 1 + right)]
            filtered.append(measure(np.vstack((l, m, r)), axis=0))
        return np.array(filtered)

    if isinstance(X, list):
        return [transform(x) for x in X]
    elif isinstance(X, np.ndarray):
        return transform(X)