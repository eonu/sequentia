import random
import numpy as np
from fastdtw import fastdtw
from collections import Counter
from scipy.spatial.distance import euclidean
from typing import Callable, Union, List, Tuple
from ...internals import Validator

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
        self._val = Validator()
        self._k = self._val.restricted_integer(
            k, lambda x: x > 0, desc='number of neighbors', expected='greater than zero')
        self._radius = self._val.restricted_integer(
            radius, lambda x: x > 0, desc='radius parameter', expected='greater than zero')
        self._distance = distance

    def fit(self, X: List[np.ndarray], y: List[str]) -> None:
        """Fits the classifier by adding labeled training observation sequences.

        Parameters:
            X {list(numpy.ndarray)} - A list of multiple observation sequences.
            y {list(str)} - A list of labels for the observation sequences.
        """
        self._X, self._y = self._val.observation_sequences_and_labels(X, y)

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

        self._val.observation_sequences(X, allow_single=True)

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
        self._val.observation_sequences_and_labels(X, y)
        self._val.one_of(metric, ['accuracy', 'f1'], desc='metric')

        if labels is not None:
            self._val.list_of_strings(labels, desc='confusion matrix labels')

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