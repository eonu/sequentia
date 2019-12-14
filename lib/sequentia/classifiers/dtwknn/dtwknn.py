import random
import numpy as np
from fastdtw import fastdtw
from collections import Counter
from scipy.spatial.distance import euclidean
from typing import Callable, Union, List

class DTWKNN:
    def __init__(self, k: int, radius: int = 10, distance: Callable = euclidean):
        # TODO: Validation checks
        self._k = k
        self._radius = radius
        self._distance = distance

    def fit(self, X: List[np.ndarray], y: List[str]) -> None:
        if not isinstance(X, list):
            raise TypeError('Collection of observation sequences must be a list')
        if not all(isinstance(sequence, np.ndarray) for sequence in X):
            raise TypeError('Each observation sequence must be a numpy.ndarray')
        if not all(sequence.ndim == 2 for sequence in X):
            raise ValueError('Each observation sequence must be two-dimensional')
        if not all(sequence.shape[1] == X[0].shape[1] for sequence in X):
            raise ValueError('Each observation sequence must have the same dimensionality')

        if isinstance(y, list):
            if not all(isinstance(label, str) for label in y):
                raise ValueError('Expected all labels to be strings')
        else:
            raise ValueError('Expected labels to be a list of strings')

        if not len(X) == len(y):
            raise ValueError('Expected the same number of observation sequences and labels')

        self._X = X
        self._y = y

    def predict(self, X: Union[np.ndarray, List[np.ndarray]]) -> Union[str, List[str]]:
        try:
            (self._X, self._y)
        except AttributeError:
            raise RuntimeError('The classifier needs to be fitted before predictions are made')

        if isinstance(X, np.ndarray):
            if not X.ndim == 2:
                raise ValueError('Observation sequence must be two-dimensional')
        elif isinstance(X, list):
            if not all(isinstance(sequence, np.ndarray) for sequence in X):
                raise TypeError('Each observation sequence must be a numpy.ndarray')
            if not all(sequence.ndim == 2 for sequence in X):
                raise ValueError('Each observation sequence must be two-dimensional')
            if not all(sequence.shape[1] == X[0].shape[1] for sequence in X):
                raise ValueError('Each observation sequence must have the same dimensionality')

        # Create a singleton array if predicting just one sequence
        if isinstance(X, np.ndarray):
            X = [X]

        labels = []
        distance = lambda x1, x2: fastdtw(x1, x2, radius=self._radius, dist=self._distance)

        for sequence in X:
            distances = [distance(sequence, x)[0] for x in self._X]
            idx = np.argpartition(distances, self._k)[:self._k]
            labels = [self._y[i] for i in idx]

            # Find the modal labels
            counter = Counter(labels)
            max_count = max(counter.values())
            modes = [k for k, v in counter.items() if v == max_count]

            # If there are multiple modes, randomly select one as the label
            # NOTE: Still okay if there is only one mode
            labels.append(random.choice(modes))

        if len(X) == 1:
            return labels[0]
        else:
            return np.array(labels) if isinstance(self._y, np.ndarray) else labels

    def evaluate(self, X, y, labels=None):
        # TODO: Validation
        raise NotImplementedError