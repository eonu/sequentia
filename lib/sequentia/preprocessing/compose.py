import numpy as np
from .transforms import Transform
from ..internals import _Validator

__all__ = ['Compose']

class Compose:
    """A pipeline of preprocessing transformations.

    Parameters
    ----------
    steps: array-like of Transform
        An ordered collection of preprocessing transformations.

    Examples
    --------
    >>> # Create some sample data
    >>> X = [np.random.random((20 * i, 3)) for i in range(1, 4)]
    >>> # Create the Compose object
    >>> pre = Compose([
    >>>     TrimConstants(),
    >>>     Center(),
    >>>     Standardize(),
    >>>     Filter(window_size=5, method='median'),
    >>>     Downsample(factor=5, method='decimate')
    >>> ])
    >>> # View a summary of the preprocessing steps
    >>> pre.summary()
    >>> # Transform the data applying transformations in order
    >>> X = pre(X)
    """

    def __init__(self, steps):
        self._val = _Validator()
        steps = list(self._val.is_iterable(steps, 'transformation steps'))
        if not all(isinstance(transform, Transform) for transform in steps):
            raise TypeError('Expected all transformation steps to be Transform objects')
        self.steps = steps

    def __call__(self, X):
        """Applies the preprocessing transformations to the provided input observation sequence(s).

        Parameters
        ----------
        X: numpy.ndarray (float) or list of numpy.ndarray (float)
            An individual observation sequence or a list of multiple observation sequences.

        Returns
        -------
        transformed: :class:`numpy:numpy.ndarray` (float) or list of :class:`numpy:numpy.ndarray` (float)
            The input observation sequence(s) with preprocessing transformations applied in order.
        """
        X = self._val.is_observation_sequences(X, allow_single=True)
        for step in self.steps:
            X = step(X, validate=False)
        return X

    def _fit(self, X):
        """Fit the preprocessing transformations with the provided observation sequence(s).

        Parameters
        ----------
        X: numpy.ndarray (float) or list of numpy.ndarray (float)
            An individual observation sequence or a list of multiple observation sequences.

        Returns
        -------
        transformed: :class:`numpy:numpy.ndarray` (float) or list of :class:`numpy:numpy.ndarray` (float)
            The input observation sequence(s) with preprocessing transformations applied in order.
        """
        X = self._val.is_observation_sequences(X, allow_single=True)
        for step in self.steps:
            X = step.fit_transform(X, validate=False)
        return X

    def fit(self, X):
        """Fit the preprocessing transformations with the provided observation sequence(s).

        Parameters
        ----------
        X: numpy.ndarray (float) or list of numpy.ndarray (float)
            An individual observation sequence or a list of multiple observation sequences.
        """
        self._fit(X)

    def fit_transform(self, X):
        """Fit the preprocessing transformations with the provided observation sequence(s) and transform them.

        Parameters
        ----------
        X: numpy.ndarray (float) or list of numpy.ndarray (float)
            An individual observation sequence or a list of multiple observation sequences.

        Returns
        -------
        transformed: :class:`numpy:numpy.ndarray` (float) or list of :class:`numpy:numpy.ndarray` (float)
            The input observation sequence(s) with preprocessing transformations applied in order.
        """
        return self._fit(X)

    def summary(self):
        """Displays an ordered summary of the preprocessing transformations."""
        if len(self.steps) == 0:
            raise RuntimeError('At least one preprocessing transformation is required')

        steps = []

        for i, step in enumerate(self.steps, start=1):
            steps.append(('{}. {}'.format(i, step._name), '   {}'.format(step)))

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