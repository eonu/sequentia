import warnings, tqdm, tqdm.auto, numpy as np, h5py
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from dtaidistance import dtw
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from ...internals import _Validator

dtw_cc = None
try:
    from dtaidistance import dtw_cc
except ImportError:
    pass

class KNNClassifier:
    """A k-Nearest Neighbor classifier that uses dynamic time warping as a distance measure for comparing observation sequences of different length.

    Parameters
    ----------
    k: int
        Number of neighbors.

    classes: Iterable[str/numeric]
        The complete set of possible classes/labels.

    weighting: 'uniform' or callable
        A callable that specifies how distance weighting should be performed.
        The callable should accept a ``numpy.ndarray`` of DTW distances, apply an element-wise weighting transformation,
        then return an equally-sized ``numpy.ndarray`` of weighted distances.

        If a `'uniform'` weighting is chosen, then the function ``lambda x: np.ones(x.size)`` is used, which weights all of the distances equally.

        If the callable is simple enough, it should be specified as a ``lambda``, but a function will also work.
        Examples of weighting functions are:

        - :math:`e^{-\\alpha x}`, specified by ``lambda x: np.exp(-alpha * x)`` for some positive :math:`\\alpha`/``alpha``,
        - :math:`\\frac{1}{x}`, specified by ``lambda x: 1 / x``.

        A good weighting function should *ideally* be defined at :math:`x=0` in the rare event that two observations are perfectly aligned and therefore have zero DTW distance.

        .. tip::
            It may be desirable to restrict DTW distances to a small range if you intend to use a weighting function.

            Using the :class:`~MinMaxScale` or :class:`~Standardize` preprocessing transformations to scale your features helps to ensure that DTW distances remain small.

    window: int > 0, optional
        The width of the Sakoe-Chiba band global constraint window.
        A larger constraint will speed up the DTW alignment by restricting the maximum temporal deviation from the diagonal of the DTW matrix.

        If no argument is provided, then no global constraint will be applied while computing the DTW matrix.

    use_c: bool
        Whether or not to use fast pure C compiled functions (from the `dtaidistance <https://github.com/wannesm/dtaidistance>`_ package) to perform the DTW computations.

        .. tip::
            If you set ``use_c = True`` and are receiving an error about a C library not being available, try reinstalling ``dtaidistance`` and disabling the cache:

            .. code-block:: console

                pip install -vvv --upgrade --no-cache-dir --force-reinstall dtaidistance

    random_state: numpy.random.RandomState, int, optional
        A random state object or seed for reproducible randomness.
    """
    def __init__(self, k, classes, weighting='uniform', window=None, use_c=False, random_state=None):
        self._val = _Validator()
        self._k = self._val.restricted_integer(
            k, lambda x: x > 0, desc='number of neighbors', expected='greater than zero')

        self._window = window if window is None else self._val.restricted_integer(
            window, lambda x: x > 0, desc='Sakoe-Chiba band width', expected='greater than zero')
        self._random_state = self._val.random_state(random_state)

        self.iterable(classes, 'classes')
        self.string_or_numeric(classes[0], 'each class')
        if all(isinstance(label, type(y[0])) for label in y[1:]):
            self._encoder = LabelEncoder().fit(classes)
        else:
            raise TypeError('Expected all classes to be of the same type')

        if weighting == 'uniform':
            self._weighting = lambda x: np.ones(x.size)
        else:
            self._val.func(weighting, 'distance weighting function')
            try:
                if isinstance(weighting(np.ones(5)), np.ndarray):
                    self._weighting = weighting
                else:
                    raise TypeError('Expected weighting function to accept a numpy.ndarray and return an equally-sized numpy.ndarray')
            except:
                raise TypeError('Expected weighting function to accept a numpy.ndarray and return an equally-sized numpy.ndarray')

        self._use_c = self._val.boolean(use_c, desc='whether or not to use fast pure C compiled functions')
        if self._use_c and (dtw_cc is None):
            warnings.warn('DTAIDistance C library not available – using Python implementation', ImportWarning)
            self._use_c = False

    def fit(self, X, y):
        """Fits the classifier by adding labeled training observation sequences.

        Parameters
        ----------
        X: List[numpy.ndarray]
            A list of multiple observation sequences.

        y: Iterable[str/numeric]
            An iterable of labels for the observation sequences.
        """
        X, y = self._val.observation_sequences_and_labels(X, y)
        self._X, self._y = X, self._encoder.transform(y)
        self._n_features = X[0].shape[1]

    def predict(self, X, verbose=True, original_labels=True, n_jobs=1):
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
        prediction(s): str/numeric or numpy.ndarray[str/numeric]
            The predicted label(s) for the observation sequence(s). If ``original_labels`` is true, then the returned labels are
            inverse-transformed into their original encoding.
        """
        try:
            (self._X, self._y)
        except AttributeError:
            raise RuntimeError('The classifier needs to be fitted before predictions are made')

        X = self._val.observation_sequences(X, allow_single=True)
        self._val.boolean(verbose, desc='verbose')
        self._val.restricted_integer(n_jobs, lambda x: x == -1 or x > 0, 'number of jobs', '-1 or greater than zero')

        if isinstance(X, np.ndarray):
            distances = np.array([self._dtw(X, x) for x in tqdm.auto.tqdm(self._X, desc='Calculating distances', disable=not(verbose))])
            return self._output(self._find_nearest(distances), original_labels)
        else:
            if n_jobs == 1:
                labels = np.zeros(len(X), dtype=int)
                for i, sequence in enumerate(tqdm.auto.tqdm(X, desc='Classifying examples', disable=not(verbose))):
                    distances = np.array([self._dtw(sequence, x) for x in self._X])
                    labels[i] = self._find_nearest(distances)
                return self._output(labels, original_labels)
            else:
                n_jobs = cpu_count() if n_jobs == -1 else n_jobs
                X_chunks = [list(chunk) for chunk in np.array_split(X, n_jobs)]
                labels = Parallel(n_jobs=n_jobs)(delayed(self._parallel_predict)(i+1, chunk, verbose) for i, chunk in enumerate(X_chunks))
                return self._output(np.concatenate(labels), original_labels) # Flatten the resulting array

    def _dtw_1d(self, a, b): # Requires fit
        """Computes the DTW distance between two univariate sequences."""
        return dtw.distance(a, b, use_c=self._use_c, window=self._window)

    def _dtw(self, A, B): # Requires fit
        """Computes the multivariate DTW distance as an average of the pairwise per-feature DTW distances."""
        return np.mean([self._dtw_1d(A[:, i], B[:, i]) for i in range(self._n_features)])

    def _argmax(self, a):
        """Same as numpy.argmax but returns all occurrences of the maximum, and is O(n) instead of O(2n).
        From: https://stackoverflow.com/a/58652335
        """
        all_, max_ = [0], a[0]
        for i in range(1, len(a)):
            if a[i] > max_:
                all_, max_ = [i], a[i]
            elif a[i] == max_:
                all_.append(i)
        return np.array(all_)

    def _find_nearest(self, distances): # Requires fit
        """Returns the mode label of the k nearest neighbors.
        Vectorization from: https://stackoverflow.com/a/49239335
        """
        # Find the indices, labels and distances of the k-nearest neighbours
        idx = np.argpartition(distances, self._k)[:self._k]
        nearest_labels = self._y[idx]
        nearest_distances = self._weighting(distances[idx])
        # Combine labels and distances into one array and sort by label
        labels_distances = np.vstack((nearest_labels, nearest_distances))
        labels_distances = labels_distances[:, labels_distances[0, :].argsort()]
        # Find indices where the label changes
        i = np.nonzero(np.diff(labels_distances[0, :]))[0] + 1
        i = np.insert(i, 0, 0)
        # Add-reduce weighted distances within each label group (ordered by label)
        label_scores = np.add.reduceat(labels_distances[1, :], i)
        # Find the mode labels (set of labels with labels scores equal to the maximum)
        max_labels = nearest_labels[self._argmax(label_scores)]
        # Randomly pick from the set of labels with the maximum label score
        return self._random_state.choice(max_labels)

    def _parallel_predict(self, process, chunk, verbose): # Requires fit
        """TODO"""
        labels = np.zeros(len(chunk), dtype=int)
        for i, sequence in enumerate(tqdm.tqdm(chunk,
            desc='Classifying examples (process {})'.format(process),
            disable=not(verbose), position=process-1)
        ):
            distances = np.array([self._dtw(sequence, x) for x in self._X])
            labels[i] = self._find_nearest(distances)
        return labels

    def _output(self, out, original_labels):
        """TODO"""
        if isinstance(out, np.ndarray):
            return self._encoder.inverse_transform(out) if original_labels else out
        else:
            return self._encoder.inverse_transform([out]).item() if original_labels else out

    def evaluate(self, X, y, verbose=True, n_jobs=1):
        """Evaluates the performance of the classifier on a batch of observation sequences and their labels.

        Parameters
        ----------
        X: List[numpy.ndarray]
            A list of multiple observation sequences.

        y: Iterable[str/numeric]
            An iterable of labels for the observation sequences.

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
        predictions = self.predict(X, verbose=verbose, original_labels=False, n_jobs=n_jobs)
        cm = confusion_matrix(self._encoder.transform(y), predictions, labels=self._encoder.transform(self._encoder.classes_))
        return np.sum(np.diag(cm)) / np.sum(cm), cm

    # def save(self, path):
    #     """Stores the :class:`KNNClassifier` object into a `HDF5 <https://support.hdfgroup.org/HDF5/doc/H5.intro.html>`_ file.

    #     .. note:
    #         As :math:`k`-NN is a non-parametric classification algorithms, saving the classifier simply saves
    #         all of the training observation sequences and labels (along with the hyper-parameters).

    #     Parameters
    #     ----------
    #     path: str
    #         File path (with or without `.h5` extension) to store the HDF5-serialized :class:`KNNClassifier` object.
    #     """

    #     try:
    #         (self._X, self._y)
    #     except AttributeError:
    #         raise RuntimeError('The classifier needs to be fitted before it can be saved')

    #     with h5py.File(path, 'w') as f:
    #         # Store hyper-parameters (k, radius)
    #         params = f.create_group('params')
    #         params.create_dataset('k', data=self._k)
    #         params.create_dataset('radius', data=self._radius)

    #         # Store training data and labels (X, y)
    #         data = f.create_group('data')
    #         X = data.create_group('X')
    #         for i, x in enumerate(self._X):
    #             X.create_dataset(str(i), data=x)
    #         data.create_dataset('y', data=np.string_(self._y))

    # @classmethod
    # def load(cls, path, encoding='utf-8', metric=euclidean, weighting=(lambda x: 1)):
    #     """Deserializes a HDF5-serialized :class:`KNNClassifier` object.

    #     Parameters
    #     ----------
    #     path: str
    #         File path of the serialized HDF5 data generated by the :meth:`save` method.

    #     encoding: str
    #         The encoding used to represent training labels when decoding the HDF5 file.

    #         .. note::
    #             Supported string encodings in Python can be found `here <https://docs.python.org/3/library/codecs.html#standard-encodings>`_.

    #     metric: callable
    #         Distance metric for FastDTW (see :class:`KNNClassifier`).

    #     weighting: callable
    #         A function that specifies how distance weighting should be performed (see :class:`KNNClassifier`).

    #     Returns
    #     -------
    #     deserialized: :class:`KNNClassifier`
    #         The deserialized DTW :math:`k`-NN classifier object.

    #     See Also
    #     --------
    #     save: Serializes a :class:`KNNClassifier` into a HDF5 file.
    #     """

    #     with h5py.File(path, 'r') as f:
    #         # Deserialize the model hyper-parameters
    #         params = f['params']
    #         clf = cls(k=int(params['k'][()]), radius=int(params['radius'][()]), metric=metric, weighting=weighting)

    #         # Deserialize the training data and labels
    #         X, y = f['data']['X'], f['data']['y']
    #         clf._X = [np.array(X[k]) for k in sorted(X.keys(), key=lambda k: int(k))]
    #         clf._y = [label.decode(encoding) for label in y]

    #     return clf