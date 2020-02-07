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
    steps: List[Transform]
        A list of preprocessing transformations.
    """

    def __init__(self, steps):
        if not (isinstance(steps, list) and all(isinstance(step, Transform) for step in steps)):
            raise TypeError("Expected steps to be list of Transform objects")
        self._val = _Validator()
        self.steps = steps

    def transform(self, X, verbose=False):
        """Applies the preprocessing transformations to the provided input observation sequence(s).

        Parameters
        ----------
        X: numpy.ndarray or List[numpy.ndarray]
            An individual observation sequence or a list of multiple observation sequences.

        verbose: bool
            Whether or not to display a progress bar when applying transformations.

        Returns
        -------
        transformed: numpy.ndarray or List[numpy.ndarray]
            The input observation sequence(s) with preprocessing transformations applied in order.
        """
        X_t = copy(X)
        pbar = tqdm(self.steps, desc='Applying transformations', disable=not(verbose and len(self.steps) > 1), leave=True, ncols='100%')
        for step in pbar:
            pbar.set_description("Applying transformations - {}".format(step._describe()))
            X_t = step.transform(X_t, verbose=False)
        return X_t

    def _fit(self, X, verbose):
        """TODO"""
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
        X: numpy.ndarray or List[numpy.ndarray]
            An individual observation sequence or a list of multiple observation sequences.

        verbose: bool
            Whether or not to display a progress bar when fitting transformations.
        """
        self._fit(X, verbose)

    def fit_transform(self, X, verbose=False):
        """Fit the preprocessing transformations with the provided observation sequence(s) and transform them.

        Parameters
        ----------
        X: numpy.ndarray or List[numpy.ndarray]
            An individual observation sequence or a list of multiple observation sequences.

        verbose: bool
            Whether or not to display a progress bar when fitting and applying transformations.

        Returns
        -------
        transformed: numpy.ndarray or List[numpy.ndarray]
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