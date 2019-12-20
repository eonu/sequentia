import random
import numpy as np
from fastdtw import fastdtw
from collections import Counter
from scipy.spatial.distance import euclidean
from sklearn.metrics import confusion_matrix
from typing import Callable, Union, List, Tuple
from tqdm import tqdm
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

    def __init__(self, k: int, radius: int = 10, metric: Callable = euclidean):
        """
        Parameters:
            k {int} - Number of neighbors.
            radius {int} - Radius parameter for FastDTW.
                See: https://pdfs.semanticscholar.org/05a2/0cde15e172fc82f32774dd0cf4fe5827cad2.pdf
            metric {Callable} - Distance metric for FastDTW.
        """
        self._val = Validator()
        self._k = self._val.restricted_integer(
            k, lambda x: x > 0, desc='number of neighbors', expected='greater than zero')
        self._radius = self._val.restricted_integer(
            radius, lambda x: x > 0, desc='radius parameter', expected='greater than zero')
        self._metric = metric

    def fit(self, X: List[np.ndarray], y: List[str]) -> None:
        """Fits the classifier by adding labeled training observation sequences.

        Parameters:
            X {list(numpy.ndarray)} - A list of multiple observation sequences.
            y {list(str)} - A list of labels for the observation sequences.
        """
        self._X, self._y = self._val.observation_sequences_and_labels(X, y)

    def predict(self, X: Union[np.ndarray, List[np.ndarray]], verbose=True) -> Union[str, List[str]]:
        """Predicts the label for an observation sequence (or multiple sequences).

        Parameters:
            X {numpy.ndarray, list(numpy.ndarray)} - An individual observation sequence or
                a list of multiple observation sequences.
            verbose {bool} - Whether to display a progress bar or not.

        Returns {numpy.ndarray, list(numpy.ndarray)}:
            The predicted labels for the observation sequence(s).
        """
        try:
            (self._X, self._y)
        except AttributeError:
            raise RuntimeError('The classifier needs to be fitted before predictions are made')

        self._val.observation_sequences(X, allow_single=True)
        self._val.boolean(verbose, desc='verbose')

        distance = lambda x1, x2: fastdtw(x1, x2, radius=self._radius, dist=self._metric)

        if isinstance(X, np.ndarray):
            distances = [distance(X, x)[0] for x in tqdm(self._X, desc='Calculating distances', disable=not(verbose))]
            idx = np.argpartition(distances, self._k)[:self._k]
            neighbor_labels = [self._y[i] for i in idx]

            # Find the modal labels
            counter = Counter(neighbor_labels)
            max_count = max(counter.values())
            modes = [k for k, v in counter.items() if v == max_count]

            # Randomly select one of the modal labels
            return random.choice(modes)
        else:
            labels = []

            for sequence in tqdm(X, desc='Classifying examples', disable=not(verbose)):
                distances = [distance(sequence, x)[0] for x in self._X]
                idx = np.argpartition(distances, self._k)[:self._k]
                neighbor_labels = [self._y[i] for i in idx]

                # Find the modal labels
                counter = Counter(neighbor_labels)
                max_count = max(counter.values())
                modes = [k for k, v in counter.items() if v == max_count]

                # Randomly select one of the modal labels
                labels.append(random.choice(modes))

            return labels

    def evaluate(self, X: List[np.ndarray], y: List[str], labels=None, verbose=True) -> Tuple[float, np.ndarray]:
        """Evaluates the performance of the classifier on a batch of observation sequences and their labels.

        Parameters:
            X {list(numpy.ndarray)} - A list of multiple observation sequences.
            y {list(str)} - A list of labels for the observation sequences.
            labels {list(str)} - A list of labels for ordering the axes of the confusion matrix.
            verbose {bool} - Whether to display a progress bar for predictions or not.

        Return: {tuple(float, numpy.ndarray)}
            - The categorical accuracy of the classifier on the observation sequences.
            - A confusion matrix representing the discrepancy between predicted and actual labels.
        """
        self._val.observation_sequences_and_labels(X, y)
        self._val.boolean(verbose, desc='verbose')

        if labels is not None:
            self._val.list_of_strings(labels, desc='confusion matrix labels')

        # Classify each observation sequence and calculate confusion matrix
        predictions = self.predict(X)
        cm = confusion_matrix(y, predictions, labels=labels)

        return np.sum(np.diag(cm)) / np.sum(cm), cm