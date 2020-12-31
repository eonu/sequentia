import numpy as np
from copy import copy
from tqdm.auto import tqdm
from .transforms import Transform
from ..internals import _Validator

__all__ = ['Preprocess']

class Preprocess:
    """A pipeline of preprocessing transformations.

    Parameters
    ----------
    steps: array-like of Transform
        An ordered collection of preprocessing transformations.

    Examples
    --------
    >>> # Create some sample data
    >>> X = [np.random.random((20 * i, 3)) for i in range(1, 4)]
    >>> # Create the Preprocess object
    >>> pre = Preprocess([
    >>>     TrimZeros(),
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
        steps = list(self._val.iterable(steps, 'transformation steps'))
        if not all(isinstance(transform, Transform) for transform in steps):
            raise TypeError('Expected all transformation steps to be Transform objects')
        self.steps = steps

    def transform(self, X, verbose=False):
        """Applies the preprocessing transformations to the provided input observation sequence(s).

        Parameters
        ----------
        X: numpy.ndarray (float) or list of numpy.ndarray (float)
            An individual observation sequence or a list of multiple observation sequences.

        verbose: bool
            Whether or not to display a progress bar when applying transformations.

        Returns
        -------
        transformed: :class:`numpy:numpy.ndarray` (float) or list of :class:`numpy:numpy.ndarray` (float)
            The input observation sequence(s) with preprocessing transformations applied in order.
        """
        X_t = copy(X)
        pbar = tqdm(self.steps, desc='Applying transformations', disable=not(verbose and len(self.steps) > 1), leave=True, ncols='100%')
        for step in pbar:
            pbar.set_description("Applying transformations - {}".format(step._describe()))
            X_t = step.transform(X_t, verbose=False)
        return X_t

    def __call__(self, X, verbose=False):
        """Alias of the :meth:`transform` method.

        See Also
        --------
        transform: Applies the transformation.
        """
        return self.transform(X, verbose)

    def _fit(self, X, verbose):
        """Fit the preprocessing transformations with the provided observation sequence(s).

        Parameters
        ----------
        X: numpy.ndarray (float) or list of numpy.ndarray (float)
            An individual observation sequence or a list of multiple observation sequences.

        verbose: bool
            Whether or not to display a progress bar when fitting transformations.
        """
        X = self._val.observation_sequences(X, allow_single=True)
        X_t = copy(X)
        pbar = tqdm(self.steps, desc='Fitting transformations', disable=not(verbose and len(self.steps) > 1), leave=True, ncols='100%')
        for step in pbar:
            pbar.set_description("Fitting transformations - {}".format(step._describe()))
            X_t = step.fit_transform(X_t, verbose=False)
        return X_t

    def fit(self, X, verbose=False):
        """Fit the preprocessing transformations with the provided observation sequence(s).

        Parameters
        ----------
        X: numpy.ndarray (float) or list of numpy.ndarray (float)
            An individual observation sequence or a list of multiple observation sequences.

        verbose: bool
            Whether or not to display a progress bar when fitting transformations.
        """
        self._fit(X, verbose)

    def fit_transform(self, X, verbose=False):
        """Fit the preprocessing transformations with the provided observation sequence(s) and transform them.

        Parameters
        ----------
        X: numpy.ndarray (float) or list of numpy.ndarray (float)
            An individual observation sequence or a list of multiple observation sequences.

        verbose: bool
            Whether or not to display a progress bar when fitting and applying transformations.

        Returns
        -------
        transformed: :class:`numpy:numpy.ndarray` (float) or list of :class:`numpy:numpy.ndarray` (float)
            The input observation sequence(s) with preprocessing transformations applied in order.
        """
        return self._fit(X, verbose)

    def summary(self):
        """Displays an ordered summary of the preprocessing transformations."""
        if len(self.steps) == 0:
            raise RuntimeError('At least one preprocessing transformation is required')

        steps = []

        for i, step in enumerate(self.steps, start=1):
            class_name, description = step.__class__.__name__, step._describe()
            steps.append(('{}. {}'.format(i, class_name), '   {}'.format(description)))

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