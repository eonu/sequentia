import sys, numbers, numpy as np
from copy import copy
if sys.version_info >= (3, 3):
    from collections.abc import Iterable
else:
    from collections import Iterable

class _Validator:
    """Performs internal validations on various input types."""

    def observation_sequences(self, X, allow_single=False):
        """Validates observation sequence(s).

        Parameters
        ----------
        X: numpy.ndarray (float) or list of numpy.ndarray (float)
            An individual observation sequence or a list of multiple observation sequences.

        allow_single: bool
            Whether to allow an individual observation sequence.

        Returns
        -------
        X: numpy.ndarray (float) or list of numpy.ndarray (float)
            The original input observation sequence(s) if valid.
        """
        X = copy(X)
        if isinstance(X, (list, np.ndarray) if allow_single else list):
            if isinstance(X, list):
                for i, x in enumerate(X):
                    if not isinstance(x, np.ndarray):
                        raise TypeError('Each observation sequence must be a numpy.ndarray')
                    if not x.ndim <= 2:
                        raise ValueError('Each observation sequence must be at most two-dimensional')
                    x = X[i] = (x if x.ndim == 2 else np.atleast_2d(x).T).astype(float)
                    if not x.shape[1] == X[0].shape[1]:
                        raise ValueError('Each observation sequence must have the same dimensionality')
            elif isinstance(X, np.ndarray):
                if not X.ndim <= 2:
                    raise ValueError('Observation sequence must be at most two-dimensional')
                X = (X if X.ndim == 2 else np.atleast_2d(X).T).astype(float)
        else:
            if allow_single:
                raise TypeError('Expected an individual observation sequence or a list of multiple observation sequences, each of type numpy.ndarray')
            else:
                raise TypeError('Expected a list of observation sequences, each of type numpy.ndarray')
        return X

    def observation_sequences_and_labels(self, X, y):
        """Validates observation sequences and corresponding labels.

        Parameters
        ----------
        X: list of numpy.ndarray (float)
            A list of multiple observation sequences.

        y: array-like of str/numeric
            A list of labels for the observation sequences.

        Returns
        -------
        X: list of numpy.ndarray (float)
            The original input observation sequences if valid.

        y: array-like of str/numeric
            The original input labels if valid.
        """
        self.observation_sequences(X, allow_single=False)
        self.iterable(y, 'labels')
        self.string_or_numeric(y[0], 'each label')
        if not all(isinstance(label, type(y[0])) for label in y[1:]):
            raise TypeError('Expected all labels to be of the same type')
        if not len(X) == len(y):
            raise ValueError('Expected the same number of observation sequences and labels')
        return X, y

    def integer(self, item, desc):
        """Validates an integer.

        Parameters
        ----------
        item: int
            The item to validate.

        desc: str
            A description of the item being validated.

        Returns
        -------
        item: int
            The original input item if valid.
        """
        if not isinstance(item, int):
            raise TypeError("Expected {} to be an integer".format(desc))
        return item

    def string(self, item, desc):
        """Validates a string.

        Parameters
        ----------
        item: str
            The item to validate.

        desc: str
            A description of the item being validated.

        Returns
        -------
        item: str
            The original input item if valid.
        """
        if not isinstance(item, str):
            raise TypeError("Expected {} to be a string".format(desc))
        return item

    def string_or_numeric(self, item, desc):
        """Validates a string or numeric type.

        Parameters
        ----------
        item: str or numeric
            The item to validate.

        desc: str
            A description of the item being validated.

        Returns
        -------
        item: str or numeric
            The original input item if valid.
        """
        if not isinstance(item, (str, numbers.Number)):
            raise TypeError("Expected {} to be a string or numeric".format(desc))
        return item

    def boolean(self, item, desc):
        """Validates a boolean.

        Parameters
        ----------
        item: bool
            The item to validate.

        desc: str
            A description of the item being validated.

        Returns
        -------
        item: bool
            The original input item if valid.
        """
        if not isinstance(item, bool):
            raise TypeError("Expected {} to be a boolean".format(desc))
        return item

    def one_of(self, item, items, desc):
        """Validates that an item is one of some permitted values.

        Parameters
        ----------
        item: Any
            The item to validate.

        items: array-like of Any
            The list of permitted values to check against.

        desc: str
            A description of the item being validated.

        Returns
        -------
        item: Any
            The original input item if valid.
        """
        if not item in items:
            raise ValueError('Expected {} to be one of {}'.format(desc, items))
        return item

    def restricted_integer(self, item, condition, desc, expected):
        """Validates an integer and checks that it satisfies some condition.

        Parameters
        ----------
        item: int
            The item to validate.

        condition: lambda
            A condition to check the item against.

        desc: str
            A description of the item being validated.

        expected: str
            A description of the condition, or expected value.

        Returns
        -------
        item: int
            The original input item if valid.
        """
        if isinstance(item, int):
            if not condition(item):
                raise ValueError('Expected {} to be {}'.format(desc, expected))
        else:
            raise TypeError("Expected {} to be an integer".format(desc))
        return item

    def restricted_float(self, item, condition, desc, expected):
        """Validates a float and checks that it satisfies some condition.

        Parameters
        ----------
        item: float
            The item to validate.

        condition: lambda
            A condition to check the item against.

        desc: str
            A description of the item being validated.

        expected: str
            A description of the condition, or expected value.

        Returns
        -------
        item: float
            The original input item if valid.
        """
        if isinstance(item, float):
            if not condition(item):
                raise ValueError('Expected {} to be {}'.format(desc, expected))
        else:
            raise TypeError("Expected {} to be a float".format(desc))
        return item

    def random_state(self, state):
        """Validates a random state object or seed.

        Parameters
        ----------
        state: None, int, numpy.random.RandomState
            A random state object or seed.

        Returns
        -------
        state: numpy.random.RandomState
            A random state object.
        """
        if state is None:
            return np.random.RandomState(seed=None)
        elif isinstance(state, int):
            return np.random.RandomState(seed=state)
        elif isinstance(state, np.random.RandomState):
            return state
        else:
            raise TypeError('Expected random state to be of type: None, int, or numpy.random.RandomState')

    def func(self, item, desc):
        """Validates a callable.

        Parameters
        ----------
        item: callable
            The item to validate.

        desc: str
            A description of the item being validated.

        Returns
        -------
        item: callable
            The original input item if valid.
        """
        if callable(item):
            return item
        else:
            raise TypeError('Expected {} to be callable'.format(desc))

    def iterable(self, item, desc):
        """Validates an iterable.

        Parameters
        ----------
        item: iterable
            The item to validate.

        desc: str
            A description of the item being validated.

        Returns
        -------
        item: iterable
            The original input item if valid.
        """
        if isinstance(item, Iterable) and hasattr(item, '__len__'):
            return item
        else:
            raise TypeError("Expected {} to be an iterable".format(desc))