import numpy as np
from .methods import _normalize, _downsample, _fft, _filtrate
from ..internals import Validator

class Preprocess:
    """Efficiently applies multiple preprocessing transformations to the provided input observation sequence(s)."""

    def __init__(self):
        self._transforms = []
        self._val = Validator()

    def normalize(self):
        """Normalizes an observation sequence (or multiple sequences) by centering observations around the mean."""
        self._transforms.append((_normalize, {}))

    def downsample(self, n, method='decimate'):
        """Downsamples an observation sequence (or multiple sequences) by either:
            - Decimating the next :math:`n-1` observations
            - Averaging the current observation with the next :math:`n-1` observations

        Parameters
        ----------
        n: int
            Downsample factor.

        method: {'decimate', 'average'}
            The downsampling method.
        """
        self._val.restricted_integer(n, lambda x: x > 1, desc='downsample factor', expected='greater than one')
        self._val.one_of(method, ['decimate', 'average'], desc='downsampling method')
        self._transforms.append((_downsample, {'n': n, 'method': method}))

    def fft(self):
        """Applies a Discrete Fourier Transform to the input observation sequence(s)."""
        self._transforms.append((_fft, {}))

    def filtrate(self, n, method='median'):
        """Applies a median or mean filter to the input observation sequence(s).

        Parameters
        ----------
        n: int
            Window size.

        method: {'median', 'mean'}
            The filtering method.
        """
        self._val.restricted_integer(n, lambda x: x > 1, desc='window size', expected='greater than one')
        self._val.one_of(method, ['median', 'mean'], desc='filtering method')
        self._transforms.append((_filtrate, {'n': n, 'method': method}))

    def transform(self, X):
        """Applies the preprocessing transformations to the provided input observation sequence(s).

        Parameters
        ----------
        X: numpy.ndarray or List[numpy.ndarray]
            An individual observation sequence or a list of multiple observation sequences.

        Returns
        -------
        transformed: numpy.ndarray or List[numpy.ndarray]
            The input observation sequence(s) with preprocessing transformations applied in order.
        """
        self._val.observation_sequences(X, allow_single=True)

        X_transform = X
        for transform, kwargs in self._transforms:
            if transform == _downsample:
                if isinstance(X_transform, np.ndarray):
                    self._val.restricted_integer(kwargs['n'], lambda x: x <= len(X_transform),
                        desc='downsample factor', expected='no greater than the number of frames')
                else:
                    self._val.restricted_integer(kwargs['n'], lambda x: x <= min(len(x) for x in X_transform),
                        desc='downsample factor', expected='no greater than the number of frames in the shortest sequence')
            elif transform == _filtrate:
                if isinstance(X, np.ndarray):
                    self._val.restricted_integer(kwargs['n'], lambda x: x <= len(X),
                        desc='window size', expected='no greater than the number of frames')
                else:
                    self._val.restricted_integer(kwargs['n'], lambda x: x <= min(len(x) for x in X),
                        desc='window size', expected='no greater than the number of frames in the shortest sequence')
            X_transform = transform(X_transform, **kwargs)
        return X_transform