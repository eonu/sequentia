import numpy as np
from .methods import _normalize, _downsample, _fft
from ..internals import Validator

class Preprocess:
    """Efficiently applies multiple preprocessing transformations to provided input observation sequences."""

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

    def transform(self, X):
        """Applies the preprocessing transformations to the provided input observation sequence(s).

        Parameters
        ----------
        X: List[numpy.ndarray]
            A list of multiple observation sequences.

        Returns
        -------
        transformed: List[numpy.ndarray]
            The input observation sequences with preprocessing transformations applied in order.
        """
        self._val.observation_sequences(X)

        X_transform = X
        for transform, kwargs in self._transforms:
            X_transform = transform(X_transform, **kwargs)
        return X_transform