import numpy as np
from copy import copy
from tqdm.auto import tqdm
from ..internals import _Validator

__all__ = ['Transform', 'Equalize', 'TrimZeros', 'MinMaxScale', 'Center', 'Standardize', 'Downsample', 'Filter']

class Transform:
    """Base class representing a single transformation."""
    def __init__(self):
        self._val = _Validator()

    def _describe(self):
        """Description of the transformation.

        Returns
        -------
        description: str
            The description of the transformation.
        """
        raise NotImplementedError

    def transform(self, X, verbose=False):
        """Applies the transformation.

        Parameters
        ----------
        X: numpy.ndarray or List[numpy.ndarray]
            An individual observation sequence or a list of multiple observation sequences.

        verbose: bool
            Whether or not to display a progress bar when applying transformations.

        Returns
        -------
        transformed: numpy.ndarray or List[numpy.ndarray]
            The transformed input observation sequence(s).
        """
        raise NotImplementedError

    def __call__(self, X, verbose=False):
        """Alias of the :meth:`transform` method."""
        return self.transform(X, verbose)

    def _apply(self, transform, X, verbose):
        """Applies the transformation to the observation sequence(s).

        Parameters
        ----------
        transform: callable
            The transformation to apply.

        X: numpy.ndarray or List[numpy.ndarray]
            An individual observation sequence or a list of multiple observation sequences.

        Returns
        -------
        transformed: numpy.ndarray or List[numpy.ndarray]
            The transformed input observation sequence(s).
        """
        X = self._val.observation_sequences(X, allow_single=True)
        verbose = self._val.boolean(verbose, 'verbose')

        def apply_transform():
            if isinstance(X, np.ndarray):
                return transform(copy(X))
            else:
                return [transform(x) for x in tqdm(copy(X), desc=self._describe(), disable=not(verbose))]

        if self._is_fitted():
            return apply_transform()
        else:
            try:
                self.fit(X)
                return apply_transform()
            except:
                raise
            finally:
                self._unfit()

    def fit(self, X):
        """Fit the transformation on the provided observation sequence(s) (without transforming them).

        Parameters
        ----------
        X: numpy.ndarray or List[numpy.ndarray]
            An individual observation sequence or a list of multiple observation sequences.
        """
        self._val.observation_sequences(X, allow_single=True)

    def _unfit(self):
        """Unfit the transformation by resetting the parameters to their default settings."""
        pass

    def _is_fitted(self):
        """Check whether or not the transformation is fitted on some observation sequence(s).

        Returns
        -------
        fitted: bool
            Whether or not the transformation is fitted.
        """
        return False

    def fit_transform(self, X, verbose=False):
        """Fit the transformation with the provided observation sequence(s) and transform them.

        Parameters
        ----------
        X: numpy.ndarray or List[numpy.ndarray]
            An individual observation sequence or a list of multiple observation sequences.

        verbose: bool
            Whether or not to display a progress bar when fitting and applying transformations.

        Returns
        -------
        transformed: numpy.ndarray or List[numpy.ndarray]
            The transformed input observation sequence(s).
        """
        self.fit(X)
        return self.transform(X, verbose)

class Equalize(Transform):
    """Equalize all observation sequence lengths by padding or trimming zeros."""
    def __init__(self):
        super().__init__()
        self.length = None

    def fit(self, X):
        X = self._val.observation_sequences(X, allow_single=True)
        self.length = max(len(x) for x in X) if isinstance(X, list) else len(X)

    def _unfit(self):
        self.length = None

    def _is_fitted(self):
        return self.length is not None

    def _describe(self):
        return 'Equalize sequence lengths'

    def transform(self, X, verbose=False):
        def equalize(x):
            T, D = x.shape
            return np.vstack((x, np.zeros((self.length - T, D)))) if T <= self.length else x[:self.length]
        return self._apply(equalize, X, verbose)

class TrimZeros(Transform):
    """Trim zero-observations from the input observation sequence(s).

    Examples
    --------
    >>> # Create some sample data
    >>> z = np.zeros((4, 3))
    >>> x = lambda i: np.vstack((z, np.random.random((10 * i, 3)), z))
    >>> X = [x(i) for i in range(1, 4)]
    >>> # Zero-trim the data
    >>> X = TrimZeros()(X)
    """

    def _describe(self):
        return 'Remove zero-observations'

    def transform(self, X, verbose=False):
        def trim_zeros(x):
            return x[~np.all(x == 0, axis=1)]
        return self._apply(trim_zeros, X, verbose)

class MinMaxScale(Transform):
    """Scales the observation sequence features to each be within a provided range.

    Parameters
    ----------
    scale: tuple(int, int)
        The range of the transformed observation sequence features.

    independent: bool
        Whether to independently compute the minimum and maximum to scale each observation sequence.
    """
    def __init__(self, scale=(0, 1), independent=True):
        super().__init__()
        if not isinstance(scale, tuple):
            raise TypeError('TODO')
        if not all(isinstance(val, int) for val in scale):
            raise TypeError('TODO')
        if not scale[0] < scale[1]:
            raise ValueError('TODO')
        self.scale = scale
        self.independent = self._val.boolean(independent, 'independent')
        if not self.independent:
            self.min, self.max = None, None

    def fit(self, X):
        X = self._val.observation_sequences(X, allow_single=True)
        if not self.independent:
            X_concat = np.vstack(X) if isinstance(X, list) else X
            self.min, self.max = X_concat.min(axis=0), X_concat.max(axis=0)

    def _unfit(self):
        if not self.independent:
            self.min, self.max = None, None

    def _is_fitted(self):
        return False if self.independent else (self.min is not None) and (self.max is not None)

    def _describe(self):
        if not self.independent:
            return 'Min-max scaling into range {}'.format(self.scale)
        else:
            return 'Min-max scaling into range {} (independent)'.format(self.scale)

    def transform(self, X, verbose=False):
        if not self.independent:
            def min_max_scale(x):
                min_, max_ = self.scale
                scale = (max_ - min_) / (self.max - self.min)
                return scale * x + min_ - self.min * scale
        else:
            def min_max_scale(x):
                min_, max_ = self.scale
                scale = (max_ - min_) / (x.max(axis=0) - x.min(axis=0))
                return scale * x + min_ - x.min(axis=0) * scale
        return self._apply(min_max_scale, X, verbose)

class Center(Transform):
    """Centers the observation sequence features around their means. Results in zero-mean features.

    Parameters
    ----------
    independent: bool
        Whether to independently compute the mean to scale each observation sequence.

    Examples
    --------
    >>> # Create some sample data
    >>> X = [np.random.random((10 * i, 3)) for i in range(1, 4)]
    >>> # Center the data
    >>> X = Center()(X)
    """
    def __init__(self, independent=True):
        super().__init__()
        self.independent = self._val.boolean(independent, 'independent')
        if not self.independent:
            self.mean = None

    def fit(self, X):
        X = self._val.observation_sequences(X, allow_single=True)
        if not self.independent:
            X_concat = np.vstack(X) if isinstance(X, list) else X
            self.mean = X_concat.mean(axis=0)

    def _unfit(self):
        if not self.independent:
            self.mean = None

    def _is_fitted(self):
        return False if self.independent else (self.mean is not None)

    def _describe(self):
        if not self.independent:
            return 'Centering around mean (zero mean)'
        else:
            return 'Centering around mean (zero mean) (independent)'

    def transform(self, X, verbose=False):
        if not self.independent:
            def center(x):
                return x - self.mean
        else:
            def center(x):
                return x - x.mean(axis=0)
        return self._apply(center, X, verbose)

class Standardize(Transform):
    """Centers the observation sequence features around their means, then scales them by their deviations.
    Results in zero-mean, unit-variance features.

    Parameters
    ----------
    independent: bool
        Whether to independently compute the mean and standard deviation to scale each observation sequence.

    Examples
    --------
    >>> # Create some sample data
    >>> X = [np.random.random((10 * i, 3)) for i in range(1, 4)]
    >>> # Standardize the data
    >>> X = Standardize()(X)
    """
    def __init__(self, independent=True):
        super().__init__()
        self.independent = self._val.boolean(independent, 'independent')
        if not self.independent:
            self.mean, self.std = None, None

    def fit(self, X):
        X = self._val.observation_sequences(X, allow_single=True)
        if not self.independent:
            X_concat = np.vstack(X) if isinstance(X, list) else X
            self.mean, self.std = X_concat.mean(axis=0), X_concat.std(axis=0)

    def _unfit(self):
        if not self.independent:
            self.mean, self.std = None, None

    def _is_fitted(self):
        return False if self.independent else (self.mean is not None) and (self.std is not None)

    def _describe(self):
        if not self.independent:
            return 'Standard scaling (zero mean, unit variance)'
        else:
            return 'Standard scaling (zero mean, unit variance) (independent)'

    def transform(self, X, verbose=False):
        if not self.independent:
            def standardize(x):
                return (x - self.mean) / self.std
        else:
            def standardize(x):
                return (x - x.mean(axis=0)) / x.std(axis=0)
        return self._apply(standardize, X, verbose)

class Downsample(Transform):
    """Downsamples an observation sequence (or multiple sequences) by either:

    - Decimating the next :math:`n-1` observations
    - Averaging the current observation with the next :math:`n-1` observations

    Parameters
    ----------
    factor: int > 0
        Downsample factor.

    method: {'decimate', 'mean'}
        The downsampling method.

    Examples
    --------
    >>> # Create some sample data
    >>> X = [np.random.random((10 * i, 3)) for i in range(1, 4)]
    >>> # Downsample the data with downsample factor 5 and decimation
    >>> X = Downsample(factor=5, method='decimate')(X)
    """
    def __init__(self, factor, method='decimate'):
        super().__init__()
        self.factor = self._val.restricted_integer(factor, lambda x: x > 0, desc='downsample factor', expected='positive')
        self.method = self._val.one_of(method, ['decimate', 'mean'], desc='downsampling method')

    def _describe(self):
        method = 'Decimation' if self.method == 'decimate' else 'Mean'
        return '{} downsampling with factor {}'.format(method, self.factor)

    def transform(self, X, verbose=False):
        X = self._val.observation_sequences(X, allow_single=True)

        if isinstance(X, np.ndarray):
            self._val.restricted_integer(self.factor, lambda x: x <= len(X),
                desc='downsample factor', expected='no greater than the number of frames')
        else:
            self._val.restricted_integer(self.factor, lambda x: x <= min(len(x) for x in X),
                desc='downsample factor', expected='no greater than the number of frames in the shortest sequence')

        def downsample(x):
            N, D = x.shape
            if self.method == 'decimate':
                return np.delete(x, [i for i in range(N) if i % self.factor != 0], 0)
            elif self.method == 'mean':
                r = len(x) % self.factor
                xn, xr = (x, None) if r == 0 else (x[:-r], x[-r:])
                dxn = xn.T.reshape(-1, self.factor).mean(axis=1).reshape(D, -1).T
                return dxn if xr is None else np.vstack((dxn, xr.mean(axis=0)))

        return self._apply(downsample, X, verbose)

class Filter(Transform):
    """Applies a median or mean filter to the input observation sequence(s).

    Parameters
    ----------
    window_size: int
        The size of the filtering window.

    method: {'median', 'mean'}
        The filtering method.

    Examples
    --------
    >>> # Create some sample data
    >>> X = [np.random.random((10 * i, 3)) for i in range(1, 4)]
    >>> # Filter the data with window size 5 and median filtering
    >>> X = Filter(window_size=5, method='median')(X)
    """
    def __init__(self, window_size, method='median'):
        super().__init__()
        self.window_size = self._val.restricted_integer(window_size, lambda x: x > 0, desc='window size', expected='positive')
        self.method = self._val.one_of(method, ['median', 'mean'], desc='filtering method')

    def _describe(self):
        return '{} filtering with window-size {}'.format(self.method.capitalize(), self.window_size)

    def transform(self, X, verbose=False):
        X = self._val.observation_sequences(X, allow_single=True)

        if isinstance(X, np.ndarray):
            self._val.restricted_integer(self.window_size, lambda x: x <= len(X),
                desc='window size', expected='no greater than the number of frames')
        else:
            self._val.restricted_integer(self.window_size, lambda x: x <= min(len(x) for x in X),
                desc='window size', expected='no greater than the number of frames in the shortest sequence')

        def filter_(x):
            measure = np.median if self.method == 'median' else np.mean
            filtered = []
            right = self.window_size // 2
            left = (self.window_size - 1) - right
            for i in range(len(x)):
                l, m, r = x[((i - left) * (left < i)):i], x[i], x[(i + 1):(i + 1 + right)]
                filtered.append(measure(np.vstack((l, m, r)), axis=0))
            return np.array(filtered)

        return self._apply(filter_, X, verbose)