import numpy as np
from ..internals import _Validator

__all__ = ['Transform', 'Custom', 'TrimConstants', 'MinMaxScale', 'Center', 'Standardize', 'Downsample', 'Filter']

class Transform:
    """Base class representing a single transformation."""

    def __init__(self):
        self._val = _Validator()
        self._name = self.__class__.__name__

    def __call__(self, X, validate=True):
        """Applies the transformation to the observation sequence(s).

        Parameters
        ----------
        X: numpy.ndarray (float) or list of numpy.ndarray (float)
            An individual observation sequence or a list of multiple observation sequences.

        validate: bool
            Whether or not to validate the input sequences.

        Returns
        -------
        transformed: :class:`numpy:numpy.ndarray` (float) or list of :class:`numpy:numpy.ndarray` (float)
            The transformed input observation sequence(s).
        """
        if self._val.is_boolean(validate, 'validate'):
            X = self._val.is_observation_sequences(X, allow_single=True)

        if self.is_fitted():
            return self._apply(X)

        try:
            self.fit(X, validate=validate)
            return self._apply(X)
        except:
            raise
        finally:
            self.unfit()

    def transform(self, x):
        """Applies the transformation to a single observation sequence.

        Parameters
        ----------
        X: numpy.ndarray (float)
            An individual observation sequence.

        Returns
        -------
        transformed: :class:`numpy:numpy.ndarray` (float)
            The transformed input observation sequence.
        """
        raise NotImplementedError

    def fit(self, X, validate=True):
        """Fit the transformation on the provided observation sequence(s) (without transforming them).

        Parameters
        ----------
        X: numpy.ndarray (float) or list of numpy.ndarray (float)
            An individual observation sequence or a list of multiple observation sequences.

        validate: bool
            Whether or not to validate the input sequences.
        """
        self._val.is_boolean(validate, 'validate')

    def fit_transform(self, X, validate=True):
        """Fit the transformation with the provided observation sequence(s) and transform them.

        Parameters
        ----------
        X: numpy.ndarray (float) or list of numpy.ndarray (float)
            An individual observation sequence or a list of multiple observation sequences.

        validate: bool
            Whether or not to validate the input sequences.

        Returns
        -------
        transformed: :class:`numpy:numpy.ndarray` (float) or list of :class:`numpy:numpy.ndarray` (float)
            The transformed input observation sequence(s).
        """
        self.fit(X, validate=validate)
        return self.__call__(X, validate=validate)

    def is_fitted(self):
        """Check whether or not the transformation is fitted on some observation sequence(s).

        Returns
        -------
        fitted: bool
            Whether or not the transformation is fitted.
        """
        return False

    def unfit(self):
        """Unfit the transformation by resetting the parameters to their default settings."""
        pass

    def _apply(self, X):
        """Applies the transformation to the observation sequence(s) (internal).

        Parameters
        ----------
        X: numpy.ndarray (float) or list of numpy.ndarray (float)
            An individual observation sequence or a list of multiple observation sequences.

        Returns
        -------
        transformed: :class:`numpy:numpy.ndarray` (float) or list of :class:`numpy:numpy.ndarray` (float)
            The transformed input observation sequence(s).
        """
        return self.transform(X) if isinstance(X, np.ndarray) else [self.transform(x) for x in X]

    def __str__(self):
        """Description of the transformation.

        Returns
        -------
        description: str
            The description of the transformation.
        """
        raise NotImplementedError

class Custom(Transform):
    """Apply a custom transformation to the input observation sequence(s).

    Parameters
    ----------
    func: callable
        A lambda or function that specifies the transformation that should be applied to a **single** observation sequence.

    name: str
        Name of the transformation.

    desc: str
        Description of the transformation.

    Examples
    --------
    >>> # Create some sample data
    >>> X = [np.random.random((10 * i, 3)) for i in range(1, 4)]
    >>> # Apply a custom transformation
    >>> X = Custom(lambda x: x**2, name='Square', desc='Square observations element-wise')(X)
    """

    def __init__(self, func, name=None, desc=None):
        super().__init__()
        self.transform = self._val.is_func(func, 'transformation')
        self._name = 'Custom' + ('' if name is None else ' ({})'.format(self._val.is_string(name, 'name')))
        self._desc = 'Apply a custom transformation' if desc is None else self._val.is_string(desc, 'description')

    def __str__(self):
        return self._desc

class TrimConstants(Transform):
    """Trim constant observations from the input observation sequence(s).

    Parameters
    ----------
    const: float
        The constant value.

    Examples
    --------
    >>> # Create some sample data
    >>> z = np.zeros((4, 3))
    >>> x = lambda i: np.vstack((z, np.random.random((10 * i, 3)), z))
    >>> X = [x(i) for i in range(1, 4)]
    >>> # Trim the data
    >>> X = TrimConstants()(X)
    """

    def __init__(self, constant=0):
        super().__init__()
        try:
            self.constant = float(constant)
        except ValueError:
            raise TypeError('Expected constant to be a float or convertible to a float')

    def transform(self, x):
        return x[~np.all(x == self.constant, axis=1)]

    def __str__(self):
        return 'Remove constant observations (={:.3})'.format(self.constant)

class MinMaxScale(Transform):
    """Scales the observation sequence features to each be within a provided range.

    Parameters
    ----------
    scale: tuple(int/float, int/float)
        The range of the transformed observation sequence features.

    independent: bool
        Whether to independently compute the minimum and maximum to scale each observation sequence.
    """

    def __init__(self, scale=(0., 1.), independent=True):
        super().__init__()
        if not isinstance(scale, tuple):
            raise TypeError('Expected scaling range to be a tuple')
        if not all(isinstance(val, (int, float)) for val in scale):
            raise TypeError('Expected upper and lower bounds of scaling range to be floats')
        if not scale[0] < scale[1]:
            raise ValueError('Expected lower bound of scaling range to be less than the upper bound')
        self.scale = scale
        self.independent = self._val.is_boolean(independent, 'independent')
        self._type = (_MinMaxScaleIndependent if independent else _MinMaxScaleAll)(scale)

    def fit(self, X, validate=True):
        super().fit(X, validate=validate)
        self._type.fit(X, validate=validate)

    def transform(self, x):
        return self._type.transform(x)

    def is_fitted(self):
        return self._type.is_fitted()

    def unfit(self):
        self._type.unfit()

    def __str__(self):
        return str(self._type)

class _MinMaxScaleAll(Transform):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        self.unfit()

    def fit(self, X, validate):
        if validate:
            X = self._val.is_observation_sequences(X, allow_single=True)
        if isinstance(X, list):
            X = np.vstack(X)
        self.min, self.max = X.min(axis=0), X.max(axis=0)

    def transform(self, x):
        min_, max_ = self.scale
        scale = (max_ - min_) / (self.max - self.min)
        return scale * x + min_ - self.min * scale

    def is_fitted(self):
        return (self.min is not None) and (self.max is not None)

    def unfit(self):
        self.min, self.max = None, None

    def __str__(self):
        return 'Min-max scaling into range {} (all)'.format(self.scale)

class _MinMaxScaleIndependent(Transform):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def transform(self, x):
        min_, max_ = self.scale
        scale = (max_ - min_) / (x.max(axis=0) - x.min(axis=0))
        return scale * x + min_ - x.min(axis=0) * scale

    def __str__(self):
        return 'Min-max scaling into range {} (independent)'.format(self.scale)

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
        self.independent = self._val.is_boolean(independent, 'independent')
        self._type = (_CenterIndependent if independent else _CenterAll)()

    def fit(self, X, validate=True):
        super().fit(X, validate=validate)
        self._type.fit(X, validate=validate)

    def transform(self, x):
        return self._type.transform(x)

    def is_fitted(self):
        return self._type.is_fitted()

    def unfit(self):
        self._type.unfit()

    def __str__(self):
        return str(self._type)

class _CenterAll(Transform):
    def __init__(self):
        super().__init__()
        self.unfit()

    def fit(self, X, validate):
        if validate:
            X = self._val.is_observation_sequences(X, allow_single=True)
        if isinstance(X, list):
            X = np.vstack(X)
        self.mean = X.mean(axis=0)

    def transform(self, x):
        return x - self.mean

    def is_fitted(self):
        return self.mean is not None

    def unfit(self):
        self.mean = None

    def __str__(self):
        return 'Centering around mean (zero mean) (all)'

class _CenterIndependent(Transform):
    def transform(self, x):
        return x - x.mean(axis=0)

    def __str__(self):
        return 'Centering around mean (zero mean) (independent)'

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
        self.independent = self._val.is_boolean(independent, 'independent')
        self._type = (_StandardizeIndependent if independent else _StandardizeAll)()

    def fit(self, X, validate=True):
        super().fit(X, validate=validate)
        self._type.fit(X, validate=validate)

    def transform(self, x):
        return self._type.transform(x)

    def is_fitted(self):
        return self._type.is_fitted()

    def unfit(self):
        self._type.unfit()

    def __str__(self):
        return str(self._type)

class _StandardizeAll(Transform):
    def __init__(self):
        super().__init__()
        self.unfit()

    def fit(self, X, validate):
        if validate:
            X = self._val.is_observation_sequences(X, allow_single=True)
        if isinstance(X, list):
            X = np.vstack(X)
        self.mean, self.std = X.mean(axis=0), X.std(axis=0)

    def transform(self, x):
        return (x - self.mean) / self.std

    def is_fitted(self):
        return (self.mean is not None) and (self.std is not None)

    def unfit(self):
        self.mean, self.std = None, None

    def __str__(self):
        return 'Standard scaling (zero mean, unit variance) (all)'

class _StandardizeIndependent(Transform):
    def transform(self, x):
        return (x - x.mean(axis=0)) / x.std(axis=0)

    def __str__(self):
        return 'Standard scaling (zero mean, unit variance) (independent)'

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
        self.factor = self._val.is_restricted_integer(factor, lambda x: x > 0, desc='downsample factor', expected='positive')
        self.method = self._val.is_one_of(method, ['decimate', 'mean'], desc='downsampling method')
        self._type = (_DownsampleDecimate if method == 'decimate' else _DownsampleMean)(factor)

    def transform(self, x):
        return self._type.transform(x)

    def _apply(self, X):
        if isinstance(X, np.ndarray):
            self._val.is_restricted_integer(self.factor, lambda x: x <= len(X),
                desc='downsample factor', expected='no greater than the number of frames')
        else:
            self._val.is_restricted_integer(self.factor, lambda x: x <= min(len(x) for x in X),
                desc='downsample factor', expected='no greater than the number of frames in the shortest sequence')
        return super()._apply(X)

    def __str__(self):
        return str(self._type)

class _DownsampleDecimate(Transform):
    def __init__(self, factor):
        self.factor = factor

    def transform(self, x):
        return np.delete(x, [i for i in range(len(x)) if i % self.factor != 0], 0)

    def __str__(self):
        return 'Decimation downsampling with factor {}'.format(self.factor)

class _DownsampleMean(Transform):
    def __init__(self, factor):
        self.factor = factor

    def transform(self, x):
        N, D = x.shape
        r = len(x) % self.factor
        xn, xr = (x, None) if r == 0 else (x[:-r], x[-r:])
        dxn = xn.T.reshape(-1, self.factor).mean(axis=1).reshape(D, -1).T
        return dxn if xr is None else np.vstack((dxn, xr.mean(axis=0)))

    def __str__(self):
        return 'Mean downsampling with factor {}'.format(self.factor)

class Filter(Transform):
    """Applies a median or mean filter to the input observation sequence(s).

    Parameters
    ----------
    window_size: int > 0
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
        self.window_size = self._val.is_restricted_integer(window_size, lambda x: x > 0, desc='window size', expected='positive')
        self.method = self._val.is_one_of(method, ['median', 'mean'], desc='filtering method')
        self._func = np.median if self.method == 'median' else np.mean

    def transform(self, x):
        filtered = []
        right = self.window_size // 2
        left = (self.window_size - 1) - right
        for i in range(len(x)):
            l, m, r = x[((i - left) * (left < i)):i], x[i], x[(i + 1):(i + 1 + right)]
            filtered.append(self._func(np.vstack((l, m, r)), axis=0))
        return np.array(filtered)

    def _apply(self, X):
        if isinstance(X, np.ndarray):
            self._val.is_restricted_integer(self.window_size, lambda x: x <= len(X),
                desc='window size', expected='no greater than the number of frames')
        else:
            self._val.is_restricted_integer(self.window_size, lambda x: x <= min(len(x) for x in X),
                desc='window size', expected='no greater than the number of frames in the shortest sequence')
        return super()._apply(X)

    def __str__(self):
        return '{} filtering with window-size {}'.format(self.method.capitalize(), self.window_size)