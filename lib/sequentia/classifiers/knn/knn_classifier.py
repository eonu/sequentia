import warnings, tqdm, tqdm.auto, numpy as np, types, pickle, marshal
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from dtaidistance import dtw, dtw_ndim
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
    k: int > 0
        Number of neighbors.

    classes: array-like of str/numeric
        The complete set of possible classes/labels.

    weighting: 'uniform' or callable
        A callable that specifies how distance weighting should be performed.
        The callable should accept a :class:`numpy:numpy.ndarray` of DTW distances, apply an element-wise weighting transformation,
        then return an equally-sized :class:`numpy:numpy.ndarray` of weighted distances.

        If a `'uniform'` weighting is chosen, then the function ``lambda x: np.ones(x.size)`` is used, which weights all of the distances equally.

        If the callable is simple enough, it should be specified as a ``lambda``, but a function will also work.
        Examples of weighting functions are:

        - :math:`e^{-\\alpha x}`, specified by ``lambda x: np.exp(-alpha * x)`` for some positive :math:`\\alpha`/``alpha``,
        - :math:`\\frac{1}{x}`, specified by ``lambda x: 1 / x``.

        A good weighting function should *ideally* be defined at :math:`x=0` in the rare event that two observations are perfectly aligned and therefore have zero DTW distance.

        .. tip::
            It may be desirable to restrict DTW distances to a small range if you intend to use a weighting function.

            Using the :class:`~MinMaxScale` or :class:`~Standardize` preprocessing transformations to scale your features helps to ensure that DTW distances remain small.

    window: 0 ≤ float ≤ 1
        The width of the Sakoe-Chiba band global constraint as a fraction of the length of the longest of the two sequences.
        A larger constraint will speed up the DTW alignment by restricting the maximum temporal deviation from the diagonal of the DTW matrix,
        but too much constraint may lead to poor alignment.

        The default value of 1 corresponds to full DTW computation with no global constraint applied.

    use_c: bool
        Whether or not to use fast pure C compiled functions (from the `dtaidistance <https://github.com/wannesm/dtaidistance>`_ package) to perform the DTW computations.

        .. tip::
            If you set ``use_c = True`` and are receiving an error about a C library not being available, try reinstalling ``dtaidistance`` and disabling the cache:

            .. code-block:: console

                pip install -vvv --upgrade --no-cache-dir --force-reinstall dtaidistance

    independent: bool
        Whether or not to allow features to be warped independently from each other. See `here <https://www.cs.ucr.edu/~eamonn/Multi-Dimensional_DTW_Journal.pdf>`_ for a good overview of both approaches.

    random_state: numpy.random.RandomState, int, optional
        A random state object or seed for reproducible randomness.

    Attributes
    ----------
    k: int > 0
        The number of neighbors.

    weighting: callable
        The distance weighting function.

    window: 0 ≤ float ≤ 1
        The width of the Sakoe-Chiba band global constraint as a fraction of the length of the longest of the two sequences.

    use_c: bool
        Whether or not to use fast pure C compiled functions to perform the DTW computations.

    encoder: sklearn.preprocessing.LabelEncoder
        The label encoder fitted on the set of ``classes`` provided during instantiation.

    classes: numpy.ndarray (str/numeric)
        The complete set of possible classes/labels.
    """

    def __init__(self, k, classes, weighting='uniform', window=1., use_c=False, independent=False, random_state=None):
        self._val = _Validator()
        self._k = self._val.restricted_integer(
            k, lambda x: x > 0, desc='number of neighbors', expected='greater than zero')
        self._window = float(window) if window in (0, 1) else self._val.restricted_float(
            window, lambda x: 0. <= x <= 1., desc='Sakoe-Chiba band width (fraction)', expected='between zero and one')
        self._random_state = self._val.random_state(random_state)

        self._val.iterable(classes, 'classes')
        self._val.string_or_numeric(classes[0], 'each class')
        if all(isinstance(label, type(classes[0])) for label in classes[1:]):
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

        self._independent = self._val.boolean(independent, 'independent')
        self._dtw = self._dtwi if independent else self._dtwd

    def fit(self, X, y):
        """Fits the classifier by adding labeled training observation sequences.

        Parameters
        ----------
        X: list of numpy.ndarray (float)
            A list of multiple observation sequences.

        y: array-like of str/numeric
            An iterable of labels for the observation sequences.
        """
        X, y = self._val.observation_sequences_and_labels(X, y)
        self._X, self._y = X, self._encoder.transform(y)
        self._n_features = X[0].shape[1]

    def predict(self, X, original_labels=True, verbose=True, n_jobs=1):
        """Predicts the label for an observation sequence (or multiple sequences).

        Parameters
        ----------
        X: numpy.ndarray (float) or list of numpy.ndarray (float)
            An individual observation sequence or a list of multiple observation sequences.

        original_labels: bool
            Whether to inverse-transform the labels to their original encoding.

        verbose: bool
            Whether to display a progress bar or not.

            .. note::
                If both ``verbose=True`` and ``n_jobs > 1``, then the progress bars for each process
                are always displayed in the console, regardless of where you are running this function from
                (e.g. a Jupyter notebook).

        n_jobs: int > 0 or -1
            | The number of jobs to run in parallel.
            | Setting this to -1 will use all available CPU cores.

        Returns
        -------
        prediction(s): str/numeric or :class:`numpy:numpy.ndarray` (str/numeric)
            The predicted label(s) for the observation sequence(s).

            If ``original_labels`` is true, then the returned labels are
            inverse-transformed into their original encoding.
        """
        try:
            (self._X, self._y)
        except AttributeError:
            raise RuntimeError('The classifier needs to be fitted before predictions are made')

        X = self._val.observation_sequences(X, allow_single=True)
        self._val.boolean(original_labels, desc='original_labels')
        self._val.boolean(verbose, desc='verbose')
        self._val.restricted_integer(n_jobs, lambda x: x == -1 or x > 0, 'number of jobs', '-1 or greater than zero')

        if isinstance(X, np.ndarray):
            distances = np.array([self._dtw(X, x) for x in tqdm.auto.tqdm(self._X, desc='Calculating distances', disable=not(verbose))])
            return self._output(self._find_nearest(distances), original_labels)
        else:
            n_jobs = min(cpu_count() if n_jobs == -1 else n_jobs, len(X))
            X_chunks = [list(chunk) for chunk in np.array_split(np.array(X, dtype=object), n_jobs)]
            labels = Parallel(n_jobs=n_jobs)(delayed(self._chunk_predict)(i+1, chunk, verbose) for i, chunk in enumerate(X_chunks))
            return self._output(np.concatenate(labels), original_labels) # Flatten the resulting array

    def evaluate(self, X, y, verbose=True, n_jobs=1):
        """Evaluates the performance of the classifier on a batch of observation sequences and their labels.

        Parameters
        ----------
        X: list of numpy.ndarray (float)
            A list of multiple observation sequences.

        y: array-like of str/numeric
            An iterable of labels for the observation sequences.

        verbose: bool
            Whether to display a progress bar for predictions or not.

        n_jobs: int > 0 or -1
            | The number of jobs to run in parallel.
            | Setting this to -1 will use all available CPU cores.

        Returns
        -------
        accuracy: float
            The categorical accuracy of the classifier on the observation sequences.

        confusion: :class:`numpy:numpy.ndarray` (int)
            The confusion matrix representing the discrepancy between predicted and actual labels.
        """
        X, y = self._val.observation_sequences_and_labels(X, y)
        self._val.boolean(verbose, desc='verbose')
        predictions = self.predict(X, original_labels=False, verbose=verbose, n_jobs=n_jobs)
        cm = confusion_matrix(self._encoder.transform(y), predictions, labels=self._encoder.transform(self._encoder.classes_))
        return np.sum(np.diag(cm)) / np.sum(cm), cm

    def save(self, path):
        """Serializes the :class:`KNNClassifier` object by pickling its hyper-parameters, variables and training data.

            .. warning::
                As :math:`k`-NN classifier must look through each training example during prediction,
                saving the classifier simply copies all of the training observation sequences and labels.

        Parameters
        ----------
        path: str
            File path (usually with `.pkl` extension) to store the serialized :class:`KNNClassifier` object.
        """
        try:
            (self._X, self._y)
        except AttributeError:
            raise RuntimeError('The classifier needs to be fitted before it can be saved')

        # Pickle the necessary hyper-parameters, variables and data
        with open(path, 'wb') as file:
            pickle.dump({
                'k': self._k,
                'classes': self._encoder.classes_,
                # Serialize the weighting function into a byte-string
                'weighting': marshal.dumps((self._weighting.__code__, self._weighting.__name__)),
                'window': self._window,
                'use_c': self._use_c,
                'independent': self._independent,
                'random_state': self._random_state,
                'X': self._X,
                'y': self._y,
                'n_features': self._n_features
            }, file)

    @classmethod
    def load(cls, path):
        """Deserializes a :class:`KNNClassifier` object which was serialized with the :meth:`save` function.

        Parameters
        ----------
        path: str
            File path of the serialized data generated by the :meth:`save` method.

        Returns
        -------
        deserialized: :class:`KNNClassifier`
            The deserialized DTW :math:`k`-NN classifier object.
        """
        with open(path, 'rb') as file:
            data = pickle.load(file)

            # Check deserialized object dictionary and keys
            keys = set(('k', 'classes', 'weighting', 'window', 'use_c', 'independent', 'random_state', 'X', 'y', 'n_features'))
            if not isinstance(data, dict):
                raise TypeError('Expected deserialized object to be a dictionary - make sure the object was serialized with the save() function')
            else:
                if len(set(keys) - set(data.keys())) != 0:
                    raise ValueError('Missing keys in deserialized object dictionary – make sure the object was serialized with the save() function')

            # Deserialize the weighting function
            weighting, name = marshal.loads(data['weighting'])
            weighting = types.FunctionType(weighting, globals(), name)

            # Instantiate a new KNNClassifier with the same hyper-parameters
            clf = cls(
                k=data['k'],
                classes=data['classes'],
                weighting=weighting,
                window=data['window'],
                use_c=data['use_c'],
                independent=data['independent'],
                random_state=data['random_state']
            )

            # Load the data directly
            clf._X, clf._y = data['X'], data['y']
            clf._n_features = data['n_features']

            return clf

    def _dtw_1d(self, a, b, window): # Requires fit
        """Computes the DTW distance between two univariate sequences."""
        return dtw.distance(a, b, use_c=self._use_c, window=window)

    def _dtwi(self, A, B): # Requires fit
        """Computes the multivariate DTW distance as the sum of the pairwise per-feature DTW distances, allowing each feature to be warped independently."""
        window = max(1, int(self._window * max(len(A), len(B))))
        return np.sum([self._dtw_1d(A[:, i], B[:, i], window=window) for i in range(self._n_features)])

    def _dtwd(self, A, B): # Requires fit
        """Computes the multivariate DTW distance so that the warping of the features depends on each other, by modifying the local distance measure."""
        window = max(1, int(self._window * max(len(A), len(B))))
        return dtw_ndim.distance(A, B, use_c=self._use_c, window=window)

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

    def _chunk_predict(self, process, chunk, verbose): # Requires fit
        """Makes predictions for a chunk of the observation sequences, for a given subprocess."""
        labels = np.zeros(len(chunk), dtype=int)
        for i, sequence in enumerate(tqdm.auto.tqdm(chunk,
            desc='Classifying examples (process {})'.format(process),
            disable=not(verbose), position=process-1)
        ):
            distances = np.array([self._dtw(sequence, x) for x in self._X])
            labels[i] = self._find_nearest(distances)
        return labels

    def _output(self, out, original_labels):
        """Inverse-transforms the labels if necessary, and returns them."""
        if isinstance(out, np.ndarray):
            return self._encoder.inverse_transform(out) if original_labels else out
        else:
            return self._encoder.inverse_transform([out]).item() if original_labels else out

    @property
    def k(self):
        return self._k

    @property
    def weighting(self):
        return self._weighting

    @property
    def window(self):
        return self._window

    @property
    def use_c(self):
        return self._use_c

    @property
    def encoder(self):
        return self._encoder

    @property
    def classes(self):
        return self._encoder.classes_

    @property
    def X(self):
        try:
            return self._X
        except AttributeError:
            raise RuntimeError('The classifier needs to be fitted first')

    @property
    def y(self):
        try:
            return self._y
        except AttributeError:
            raise RuntimeError('The classifier needs to be fitted first')

    def __repr__(self):
        module = self.__class__.__module__
        out = '{}{}('.format('' if module == '__main__' else '{}.'.format(module), self.__class__.__name__)
        attrs = [
            ('k', repr(self._k)),
            ('window', repr(self._window)),
            ('use_c', repr(self._use_c)),
            ('independent', repr(self._independent)),
            ('classes', repr(list(self._encoder.classes_)))
        ]
        try:
            (self._X, self._y)
            attrs.extend([('X', '[...]'), ('y', 'array([...])')])
        except AttributeError:
            pass
        return out + ', '.join('{}={}'.format(name, val) for name, val in attrs) + ')'