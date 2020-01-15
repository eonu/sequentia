import scipy.fftpack
import numpy as np
from ..internals import _Validator

def trim_zeros(X):
    """Trim zero-observations from the input observation sequence(s).

    Parameters
    ----------
    X: numpy.ndarray or List[numpy.ndarray]
        An individual observation sequence or a list of multiple observation sequences.

    Returns
    -------
    trimmed: numpy.ndarray or List[numpy.ndarray]
        The zero-trimmed input observation sequence(s).
    """
    val = _Validator()
    X = val.observation_sequences(X, allow_single=True)
    return _trim_zeros(X)

def _trim_zeros(X):
    def transform(x):
        return x[~np.all(x == 0, axis=1)]

    if isinstance(X, list):
        return [transform(x) for x in X]
    elif isinstance(X, np.ndarray):
        return transform(X)

def center(X):
    """Centers an observation sequence (or multiple sequences) by centering observations around the mean.

    Parameters
    ----------
    X: numpy.ndarray or List[numpy.ndarray]
        An individual observation sequence or a list of multiple observation sequences.

    Returns
    -------
    centered: numpy.ndarray or List[numpy.ndarray]
        The centered input observation sequence(s).
    """
    val = _Validator()
    X = val.observation_sequences(X, allow_single=True)
    return _center(X)

def _center(X):
    def transform(x):
        return x - x.mean(axis=0)

    if isinstance(X, list):
        return [transform(x) for x in X]
    elif isinstance(X, np.ndarray):
        return transform(X)

def standardize(X):
    """Standardizes an observation sequence (or multiple sequences) by transforming observations
    so that they have zero mean and unit variance.

    Parameters
    ----------
    X: numpy.ndarray or List[numpy.ndarray]
        An individual observation sequence or a list of multiple observation sequences.

    Returns
    -------
    standardized: numpy.ndarray or List[numpy.ndarray]
        The standardized input observation sequence(s).
    """
    val = _Validator()
    X = val.observation_sequences(X, allow_single=True)
    return _standardize(X)

def _standardize(X):
    def transform(x):
        return (x - x.mean(axis=0)) / x.std(axis=0)

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
    val = _Validator()
    X = val.observation_sequences(X, allow_single=True)
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
    val = _Validator()
    X = val.observation_sequences(X, allow_single=True)
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
    val = _Validator()
    X = val.observation_sequences(X, allow_single=True)
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