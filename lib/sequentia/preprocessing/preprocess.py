import numpy as np
from .methods import (
    _trim_zeros, _center, _standardize, _downsample, _fft, _filtrate
)
from ..internals import _Validator

class Preprocess:
    """Efficiently applies multiple preprocessing transformations to the provided input observation sequence(s)."""

    def __init__(self):
        self._transforms = []
        self._val = _Validator()

    def trim_zeros(self):
        """Trim zero-observations from the input observation sequence(s)."""
        self._transforms.append((_trim_zeros, {}))

    def center(self):
        """Centers an observation sequence (or multiple sequences) by centering observations around the mean."""
        self._transforms.append((_center, {}))

    def standardize(self):
        """Standardizes an observation sequence (or multiple sequences) by transforming observations
        so that they have zero mean and unit variance."""
        self._transforms.append((_standardize, {}))

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
        X_transform = self._val.observation_sequences(X, allow_single=True)
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

    def summary(self):
        """Displays an ordered summary of the preprocessing transformations."""
        if len(self._transforms) == 0:
            raise RuntimeError('At least one preprocessing transformation is required')

        steps = []

        for i, (transform, kwargs) in enumerate(self._transforms):
            idx = i + 1
            if transform == _center:
                steps.append(('{}. Centering'.format(idx), None))
            elif transform == _standardize:
                steps.append(('{}. Standardization'.format(idx), None))
            elif transform == _downsample:
                header = 'Decimation' if kwargs['method'] == 'decimate' else 'Averaging'
                steps.append((
                    '{}. Downsampling:'.format(idx),
                    '   {} with downsample factor (n={})'.format(header, kwargs['n'])
                ))
            elif transform == _fft:
                steps.append(('{}. Discrete Fourier Transform'.format(idx), None))
            elif transform == _filtrate:
                steps.append((
                    '{}. Filtering:'.format(idx),
                    '   {} filter with window size (n={})'.format(kwargs['method'].capitalize(), kwargs['n'])
                ))
            elif transform == _trim_zeros:
                steps.append(('{}. Zero-trimming'.format(idx), None))

        title = 'Preprocessing summary:'
        length = max(max(len(h), 0 if b is None else len(b)) for h, b in steps)
        length = len(title) if length < len(title) else length

        print(title.center(length, ' '))
        print('=' * length)
        for i, (head, body) in enumerate(steps):
            print(head)
            if body is not None:
                print(body)
            if i != len(steps) - 1:
                print('-' * length)
        print('=' * length)