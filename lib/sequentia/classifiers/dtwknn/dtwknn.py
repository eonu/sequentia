import random
import numpy as np
from fastdtw import fastdtw
from collections import Counter
from scipy.spatial.distance import euclidean
from typing import Callable, Union, List, Tuple

class DTWKNN:
    """A k-Nearest Neighbor classifier that compares differing length observation sequences
        using the efficient FastDTW dynamic time warping algorithm.

    Example:
        >>> import numpy as np
        >>> from sequentia.classifiers import DTWKNN
        >>> ​
        >>> # Create some sample data
        >>> X = [np.random.random((10 * i, 3)) for i in range(1, 4)]
        >>> y = ['class0', 'class1', 'class1']
        >>> ​
        >>> # Create and fit the classifier
        >>> clf = DTWKNN(k=1, radius=5)
        >>> clf.fit(X, y)
        >>> ​
        >>> # Predict labels for the training data (just as an example)
        >>> clf.predict(X)
    """

    def __init__(self, k: int, radius: int = 10, distance: Callable = euclidean):
        """
        Parameters:
            k {int} - Number of neighbors.
            radius {int} - Radius parameter for FastDTW.
                See: https://pdfs.semanticscholar.org/05a2/0cde15e172fc82f32774dd0cf4fe5827cad2.pdf
        """
        if isinstance(k, int):
            if k < 1:
                raise ValueError('Expected number of neighbors to be greater than zero')
        else:
            raise TypeError("Expected number of neighbors to be an 'int'")

        if isinstance(radius, int):
            if radius < 1:
                raise ValueError('Expected radius parameter to be greater than zero')
        else:
            raise TypeError("Expected radius parameter to be an 'int'")

        self._k = k
        self._radius = radius
        self._distance = distance

    def fit(self, X: List[np.ndarray], y: List[str]) -> None:
        """Fits the classifier by adding labeled training observation sequences.

        Parameters:
            X {list(numpy.ndarray)} - A list of multiple observation sequences.
            y {list(str)} - A list of labels for the observation sequences.
        """
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
        """Predicts the label for an observation sequence (or multiple sequences).

        Parameters:
            X {numpy.ndarray, list(numpy.ndarray)} - An individual observation sequence or
                a list of multiple observation sequences.

        Returns {numpy.ndarray, list(numpy.ndarray)}:
            The predicted labels for the observation sequence(s).
        """
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
            neighbor_labels = [self._y[i] for i in idx]

            # Find the modal labels
            counter = Counter(neighbor_labels)
            max_count = max(counter.values())
            modes = [k for k, v in counter.items() if v == max_count]

            # If there are multiple modes, randomly select one as the label
            # NOTE: Still okay if there is only one mode
            labels.append(random.choice(modes))

        return labels[0] if len(X) == 1 else labels

    def evaluate(self, X: List[np.ndarray], y: List[str], metric='accuracy', labels=None) -> Tuple[float, np.ndarray]:
        """Evaluates the performance of the classifier on a batch of observation sequences and their labels.

        Parameters:
            X {list(numpy.ndarray)} - A list of multiple observation sequences.
            y {list(str)} - A list of labels for the observation sequences.
            metric {str} - A performance metric for the classification - one of
                'accuracy' (categorical accuracy) or 'f1' (F1 score = harmonic mean of precision and recall).
            labels {list(str)} - A list of labels for ordering the axes of the confusion matrix.

        Return: {tuple(float, numpy.ndarray)}
            - The specified performance result: either categorical accuracy or F1 score.
            - A confusion matrix representing the discrepancy between predicted and actual labels.
        """
        if isinstance(X, list):
            if not all(isinstance(sequence, np.ndarray) for sequence in X):
                raise TypeError('Each observation sequence must be a numpy.ndarray')
            if not all(sequence.ndim == 2 for sequence in X):
                raise ValueError('Each observation sequence must be two-dimensional')
            if not all(sequence.shape[1] == X[0].shape[1] for sequence in X):
                raise ValueError('Each observation sequence must have the same dimensionality')
        else:
            raise TypeError('Expected a list of observation sequences, each of type numpy.ndarray')

        if isinstance(y, list):
            if not all(isinstance(label, str) for label in y):
                raise ValueError('Expected all labels to be strings')
        else:
            raise ValueError('Expected labels to be a list of strings')

        if not len(X) == len(y):
            raise ValueError('Expected the same number of observation sequences and labels')

        if metric not in ['accuracy', 'f1']:
            raise ValueError("Expected `metric` to be one of 'accuracy' or 'f1'")

        if labels is not None:
            if isinstance(labels, list):
                if not all(isinstance(label, str) for label in labels):
                    raise ValueError('Expected all confusion matrix labels to be strings')
            else:
                raise ValueError('Expected confusion matrix labels to be a list of strings')

        # Classify each observation sequence
        predictions = self.predict(X)

        # Calculate confusion matrix and precision and recall
        cm = confusion_matrix(y, predictions, labels=labels)
        precision = np.mean(np.diag(cm) / np.sum(cm, axis=0))
        recall = np.mean(np.diag(cm) / np.sum(cm, axis=1))

        if metric == 'accuracy':
            return np.sum(np.diag(cm)) / len(predictions), cm
        elif metric == 'f1':
            return 2.0 * precision * recall / (precision + recall), cm