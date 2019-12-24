import numpy as np
from typing import Union, List
from .methods import _normalize, _downsample, _fft
from ..internals import Validator

class Preprocess:
    """Efficiently applies multiple preprocessing transformations to provided input observation sequences.

    Example:
        >>> import numpy as np
        >>> from sequentia.preprocessing import Preprocess
        >>> ​
        >>> # Create some sample data
        >>> X = [np.random.random((10 * i, 3)) for i in range(1, 4)]
        >>> ​
        >>> # Create the Preprocess object
        >>> pre = Preprocess()
        >>> pre.normalize()
        >>> pre.downsample(10, method='average')
        >>> pre.fft()
        >>> ​
        >>> # Transform the data applying transformations in order
        >>> X = pre.transform(X)
    """

    def __init__(self):
        self._transforms = []
        self._val = Validator()

    def normalize(self) -> None:
        """Normalizes an observation sequence (or multiple sequences) by centering observations around the mean."""
        self._transforms.append((_normalize, {}))

    def downsample(self, n: int, method='decimate') -> None:
        """Downsamples an observation sequence (or multiple sequences) by:
            - Decimating the next n-1 observations
            - Averaging the current observation with the next n-1 observations

        Parameters:
            n {int} - Downsample factor. This downsamples the current observation
                by either decimating the next n-1 observations or computing an average with them.
            method {str} - The downsamplimg method, either 'decimate' or 'average'.
        """
        self._val.restricted_integer(n, lambda x: x > 1, desc='downsample factor', expected='greater than one')
        self._val.one_of(method, ['decimate', 'average'], desc='downsampling method')
        self._transforms.append((_downsample, {'n': n, 'method': method}))

    def fft(self) -> None:
        """Applies a Discrete Fourier Transform to the input observation sequence(s)."""
        self._transforms.append((_fft, {}))

    def transform(self, X: List[np.ndarray]) -> List[np.ndarray]:
        """Applies the preprocessing transformations to the provided input observation sequence(s).

        Parameters:
            X {list(numpy.ndarray)} - A list of multiple observation sequences.

        Returns {list(numpy.ndarray)}:
            The input observation sequences with preprocessing transformations applied in order.
        """
        self._val.observation_sequences(X)

        X_transform = X
        for transform, kwargs in self._transforms:
            X_transform = transform(X_transform, **kwargs)
        return X_transform