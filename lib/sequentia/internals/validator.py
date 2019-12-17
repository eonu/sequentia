import numpy as np

class Validator:
    """Performs internal validations on various input types."""

    def observation_sequences(self, X, allow_single=False):
        """Validates observation sequence(s).

        Parameters:
            X {np.ndarray, list(np.ndarray)} - An individual observation sequence or
                a list of multiple observation sequences.
            allow_single {bool} - Whether to allow an individual observation sequence.
        """
        if isinstance(X, (list, np.ndarray) if allow_single else list):
            if isinstance(X, list):
                if not all(isinstance(sequence, np.ndarray) for sequence in X):
                    raise TypeError('Each observation sequence must be a numpy.ndarray')
                if not all(sequence.ndim == 2 for sequence in X):
                    raise ValueError('Each observation sequence must be two-dimensional')
                if not all(sequence.shape[1] == X[0].shape[1] for sequence in X):
                    raise ValueError('Each observation sequence must have the same dimensionality')
            elif isinstance(X, np.ndarray):
                if not X.ndim == 2:
                    raise ValueError('Observation sequence must be two-dimensional')
        else:
            if allow_single:
                raise TypeError('Expected an individual observation sequence or a list of multiple observation sequences, each of type numpy.ndarray')
            else:
                raise TypeError('Expected a list of observation sequences, each of type numpy.ndarray')
        return X

    def observation_sequences_and_labels(self, X, y):
        """Validates observation sequences and corresponding labels.

        Parameters:
            X {np.ndarray, list(np.ndarray)} - An individual observation sequence or
                a list of multiple observation sequences.
            y {list(str)} - A list of labels for the observation sequences.
        """
        self.observation_sequences(X, allow_single=False)
        self.list_of_strings(y, desc='labels')
        if not len(X) == len(y):
            raise ValueError('Expected the same number of observation sequences and labels')
        return X, y

    def integer(self, item, desc):
        """Validates an integer.

        Parameters:
            item {int} - The item to validate.
            desc {str} - A description of the item being validated.
        """
        if not isinstance(item, int):
            raise TypeError("Expected {} to be an integer".format(desc))
        return item

    def string(self, item, desc):
        """Validates a string.

        Parameters:
            item {int} - The item to validate.
            desc {str} - A description of the item being validated.
        """
        if not isinstance(item, str):
            raise TypeError("Expected {} to be a string".format(desc))
        return item

    def boolean(self, item, desc):
        """Validates a boolean.

        Parameters:
            item {int} - The item to validate.
            desc {str} - A description of the item being validated.
        """
        if not isinstance(item, bool):
            raise TypeError("Expected {} to be a boolean".format(desc))
        return item

    def one_of(self, item, items, desc):
        """Validates that an item is one of some permitted values.

        Parameters:
            item {any} - The item to validate.
            items {list(any)} - The list of permitted values to check against.
            desc {str} - A description of the item being validated.
        """
        if not item in items:
            raise ValueError('Expected {} to be one of {}'.format(desc, items))
        return item

    def restricted_integer(self, item, condition, desc, expected):
        """Validates an integer and checks that it satisfies some condition.

        Parameters:
            item {int} - The item to validate.
            condition {lambda} - A condition to check the item against.
            desc {str} - A description of the item being validated.
            expected {str} - A description of the condition, or expected value.
        """
        if isinstance(item, int):
            if not condition(item):
                raise ValueError('Expected {} to be {}'.format(desc, expected))
        else:
            raise TypeError("Expected {} to be an integer".format(desc))
        return item

    def restricted_float(self, item, condition, desc, expected):
        """Validates a float and checks that it satisfies some condition.

        Parameters:
            item {float} - The item to validate.
            condition {lambda} - A condition to check the item against.
            desc {str} - A description of the item being validated.
            expected {str} - A description of the condition, or expected value.
        """
        if isinstance(item, float):
            if not condition(item):
                raise ValueError('Expected {} to be {}'.format(desc, expected))
        else:
            raise TypeError("Expected {} to be a float".format(desc))
        return item

    def list_of_strings(self, items, desc):
        """Validates a list and checks that it consists entirely of strings.

        Parameters:
            items {list(str)} - The item to validate.
            desc {str} - A description of the item being validated.
        """
        if isinstance(items, list):
            if not all(isinstance(item, str) for item in items):
                raise ValueError('Expected all {} to be strings'.format(desc))
        else:
            raise ValueError('Expected {} to be a list of strings'.format(desc))
        return items