import tqdm
import tqdm.auto
import random
import numpy as np
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from fastdtw import fastdtw
from collections import Counter
from scipy.spatial.distance import euclidean
from sklearn.metrics import confusion_matrix
from ...internals import _Validator

class DTWKNN:
    """A k-Nearest Neighbor classifier that compares differing length observation sequences using the efficient FastDTW dynamic time warping algorithm.

    Parameters
    ----------
    k: int
        Number of neighbors.

    radius: int
        Radius parameter for FastDTW.

        **See**: `Stan Salvador, and Philip Chan. "FastDTW: Toward accurate dynamic time warping in linear time and space." Intelligent Data Analysis 11.5 (2007), 561-580. <https://pdfs.semanticscholar.org/05a2/0cde15e172fc82f32774dd0cf4fe5827cad2.pdf>`_

    metric: callable
        Distance metric for FastDTW.
    """

    def __init__(self, k, radius, metric=euclidean):
        self._val = _Validator()
        self._k = self._val.restricted_integer(
            k, lambda x: x > 0, desc='number of neighbors', expected='greater than zero')
        self._radius = self._val.restricted_integer(
            radius, lambda x: x > 0, desc='radius parameter', expected='greater than zero')
        self._metric = metric

    def fit(self, X, y):
        """Fits the classifier by adding labeled training observation sequences.

        Parameters
        ----------
        X: List[numpy.ndarray]
            A list of multiple observation sequences.

        y: List[str]
            A list of labels for the observation sequences.
        """
        self._X, self._y = self._val.observation_sequences_and_labels(X, y)

    def predict(self, X, verbose=True, n_jobs=1):
        """Predicts the label for an observation sequence (or multiple sequences).

        Parameters
        ----------
        X: numpy.ndarray or List[numpy.ndarray]
            An individual observation sequence or a list of multiple observation sequences.

        verbose: bool
            Whether to display a progress bar or not.

        n_jobs: int
            | The number of jobs to run in parallel.
            | Setting this to -1 will use all available CPU cores.

        Returns
        -------
        prediction(s): str or List[str]
            The predicted label(s) for the observation sequence(s).
        """
        try:
            (self._X, self._y)
        except AttributeError:
            raise RuntimeError('The classifier needs to be fitted before predictions are made')

        X = self._val.observation_sequences(X, allow_single=True)
        self._val.boolean(verbose, desc='verbose')
        self._val.restricted_integer(n_jobs, lambda x: x == -1 or x > 0, 'number of jobs', '-1 or greater than zero')

        # FastDTW distance measure
        distance = lambda x1, x2: fastdtw(x1, x2, radius=self._radius, dist=self._metric)[0]

        def find_modes(distances):
            idx = np.argpartition(distances, self._k)[:self._k]
            neighbor_labels = [self._y[i] for i in idx]
            # Find the modal labels
            counter = Counter(neighbor_labels)
            max_count = max(counter.values())
            return [k for k, v in counter.items() if v == max_count]

        if isinstance(X, np.ndarray):
            distances = [distance(X, x) for x in tqdm.auto.tqdm(self._X, desc='Calculating distances', disable=not(verbose))]
            modes = find_modes(distances)
            # Randomly select one of the modal labels
            return random.choice(modes)
        else:
            if n_jobs == 1:
                labels = []
                for O in tqdm.auto.tqdm(X, desc='Classifying examples', disable=not(verbose)):
                    distances = [distance(O, x) for x in self._X]
                    modes = find_modes(distances)
                    # Randomly select one of the modal labels
                    labels.append(random.choice(modes))
                return labels
            else:
                def parallel_predict(process, X_chunk):
                    labels = []
                    for O in tqdm.tqdm(X_chunk, desc='Classifying examples (process {})'.format(process), disable=not(verbose), position=process-1):
                        distances = [distance(O, x) for x in self._X]
                        modes = find_modes(distances)
                        labels.append(random.choice(modes))
                    return labels

                n_jobs = cpu_count() if n_jobs == -1 else n_jobs
                X_chunks = [list(chunk) for chunk in np.array_split(X, n_jobs)]
                labels = Parallel(n_jobs=n_jobs)(delayed(parallel_predict)(i+1, chunk) for i, chunk in enumerate(X_chunks))
                return [label for sublist in labels for label in sublist] # Flatten the resulting array

    def evaluate(self, X, y, labels=None, verbose=True, n_jobs=1):
        """Evaluates the performance of the classifier on a batch of observation sequences and their labels.

        Parameters
        ----------
        X: List[numpy.ndarray]
            A list of multiple observation sequences.

        y: List[str]
            A list of labels for the observation sequences.

        labels: List[str]
            A list of labels for ordering the axes of the confusion matrix.

        verbose: bool
            Whether to display a progress bar for predictions or not.

        n_jobs: int
            | The number of jobs to run in parallel.
            | Setting this to -1 will use all available CPU cores.

        Returns
        -------
        accuracy: float
            The categorical accuracy of the classifier on the observation sequences.

        confusion: numpy.ndarray
            The confusion matrix representing the discrepancy between predicted and actual labels.
        """
        X, y = self._val.observation_sequences_and_labels(X, y)
        self._val.boolean(verbose, desc='verbose')

        if labels is not None:
            self._val.list_of_strings(labels, desc='confusion matrix labels')

        # Classify each observation sequence and calculate confusion matrix
        predictions = self.predict(X, verbose=verbose, n_jobs=n_jobs)
        cm = confusion_matrix(y, predictions, labels=labels)

        return np.sum(np.diag(cm)) / np.sum(cm), cm