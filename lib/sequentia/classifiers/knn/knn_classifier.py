import tqdm, tqdm.auto, random, numpy as np, h5py
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.metrics import confusion_matrix
from ...internals import _Validator

class KNNClassifier:
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

    weighting: callable
        A function that specifies how distance weighting should be performed. Using a constant-valued function to set all weights equally is equivalent to no weighting (which is the default configuration).
        Common weighting functions are :math:`e^{- \\alpha x}` or :math:`\\frac{1}{x}`, where :math:`x` is the DTW distance between two observation sequences.

        A weighting function should *ideally* be defined at :math:`x=0` in the rare event that two observation sequences are perfectly aligned
        (i.e. have zero DTW distance).

        - **Input**: :math:`x \geq 0`, a DTW distance between two observation sequences.
        - **Output**: A floating point value representing the weight used to perform nearest neighbor classification.

        .. note::
            Depending on your distance *metric*, it may be desirable to restrict DTW distances to a small range if you intend to use a weighting function.

            Using the :class:`~MinMaxScale` or :class:`~Standardize` preprocessing transformations to scale your features helps to ensure that distances remain small.
    """

    def __init__(self, k, radius, metric=euclidean, weighting=(lambda x: 1)):
        self._val = _Validator()
        self._k = self._val.restricted_integer(
            k, lambda x: x > 0, desc='number of neighbors', expected='greater than zero')
        self._radius = self._val.restricted_integer(
            radius, lambda x: x > 0, desc='radius parameter', expected='greater than zero')
        self._metric = self._val.func(metric, 'DTW distance metric')
        self._weighting = self._val.func(weighting, 'distance weighting function')

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

        if isinstance(X, np.ndarray):
            distances = [distance(X, x) for x in tqdm.auto.tqdm(self._X, desc='Calculating distances', disable=not(verbose))]
            return self._find_nearest(distances)
        else:
            if n_jobs == 1:
                labels = []
                for sequence in tqdm.auto.tqdm(X, desc='Classifying examples', disable=not(verbose)):
                    distances = [distance(sequence, x) for x in self._X]
                    labels.append(self._find_nearest(distances))
                return labels
            else:
                n_jobs = cpu_count() if n_jobs == -1 else n_jobs
                X_chunks = [list(chunk) for chunk in np.array_split(X, n_jobs)]
                labels = Parallel(n_jobs=n_jobs)(delayed(self._parallel_predict)(i+1, chunk, distance, verbose) for i, chunk in enumerate(X_chunks))
                return [label for sublist in labels for label in sublist] # Flatten the resulting array

    def _find_nearest(self, distances):
        # Find the indices of the k nearest points
        idx = np.argpartition(distances, self._k)[:self._k]

        # Calculate class scores by accumulating weighted distances
        class_scores = {}
        for i in idx:
            label = self._y[i]
            if label in class_scores:
                class_scores[label] += self._weighting(distances[i])
            else:
                class_scores[label] = 0

        # Find the labels with the maximum class score
        max_score = max(class_scores.values())
        max_labels = [k for k, v in class_scores.items() if v == max_score]

        # Randomly pick from the set of labels with the maximum class score
        return random.choice(max_labels)

    def _parallel_predict(self, process, chunk, distance, verbose):
        labels = []
        for sequence in tqdm.tqdm(chunk, desc='Classifying examples (process {})'.format(process), disable=not(verbose), position=process-1):
            distances = [distance(sequence, x) for x in self._X]
            labels.append(self._find_nearest(distances))
        return labels

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

    def save(self, path):
        """Stores the :class:`KNNClassifier` object into a `HDF5 <https://support.hdfgroup.org/HDF5/doc/H5.intro.html>`_ file.

        .. note:
            As :math:`k`-NN is a non-parametric classification algorithms, saving the classifier simply saves
            all of the training observation sequences and labels (along with the hyper-parameters).

        Parameters
        ----------
        path: str
            File path (with or without `.h5` extension) to store the HDF5-serialized :class:`KNNClassifier` object.
        """

        try:
            (self._X, self._y)
        except AttributeError:
            raise RuntimeError('The classifier needs to be fitted before it can be saved')

        with h5py.File(path, 'w') as f:
            # Store hyper-parameters (k, radius)
            params = f.create_group('params')
            params.create_dataset('k', data=self._k)
            params.create_dataset('radius', data=self._radius)

            # Store training data and labels (X, y)
            data = f.create_group('data')
            X = data.create_group('X')
            for i, x in enumerate(self._X):
                X.create_dataset(str(i), data=x)
            data.create_dataset('y', data=np.string_(self._y))

    @classmethod
    def load(cls, path, encoding='utf-8', metric=euclidean, weighting=(lambda x: 1)):
        """Deserializes a HDF5-serialized :class:`KNNClassifier` object.

        Parameters
        ----------
        path: str
            File path of the serialized HDF5 data generated by the :meth:`save` method.

        encoding: str
            The encoding used to represent training labels when decoding the HDF5 file.

            .. note::
                Supported string encodings in Python can be found `here <https://docs.python.org/3/library/codecs.html#standard-encodings>`_.

        metric: callable
            Distance metric for FastDTW (see :class:`KNNClassifier`).

        weighting: callable
            A function that specifies how distance weighting should be performed (see :class:`KNNClassifier`).

        Returns
        -------
        deserialized: :class:`KNNClassifier`
            The deserialized DTW :math:`k`-NN classifier object.

        See Also
        --------
        save: Serializes a :class:`KNNClassifier` into a HDF5 file.
        """

        with h5py.File(path, 'r') as f:
            # Deserialize the model hyper-parameters
            params = f['params']
            clf = cls(k=int(params['k'][()]), radius=int(params['radius'][()]), metric=metric, weighting=weighting)

            # Deserialize the training data and labels
            X, y = f['data']['X'], f['data']['y']
            clf._X = [np.array(X[k]) for k in sorted(X.keys(), key=lambda k: int(k))]
            clf._y = [label.decode(encoding) for label in y]

        return clf