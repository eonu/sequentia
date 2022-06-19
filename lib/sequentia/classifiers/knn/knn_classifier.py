import warnings, types, pickle, marshal, numpy as np
from tqdm.auto import tqdm
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

    weighting: callable, optional
        A callable that specifies how distance weighting should be performed.
        The callable should accept a :class:`numpy:numpy.ndarray` of DTW distances, apply an element-wise weighting transformation,
        then return an equally-sized :class:`numpy:numpy.ndarray` of weighted distances.

        If no weighting is chosen then the function ``lambda x: np.ones_like(x)`` is used, which weights all of the distances equally.

        Examples of weighting functions are:

        - :math:`e^{-\\alpha x}`, specified by ``lambda x: np.exp(-alpha * x)`` for some positive :math:`\\alpha`,
        - :math:`\\frac{1}{x}`, specified by ``lambda x: 1 / x``.

        A good weighting function should *ideally* be defined at :math:`x=0` in the event that two observations are perfectly aligned and therefore have zero DTW distance.

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
    k (property): int > 0
        The number of neighbors.

    weighting (property): callable
        The distance weighting function.

    window (property): 0 ≤ float ≤ 1
        The width of the Sakoe-Chiba band global constraint as a fraction of the length of the longest of the two sequences.

    use_c (property): bool
        Whether or not to use fast pure C compiled functions to perform the DTW computations.

    encoder_ (property): sklearn.preprocessing.LabelEncoder
        The label encoder fitted on the set of ``classes`` provided during instantiation.

    classes_ (property): numpy.ndarray (str/numeric)
        The complete set of possible classes/labels.

    X_ (property): list of numpy.ndarray (float)
        A list of multiple observation sequences used to fit the classifier.

    y_ (property): numpy.ndarray (int)
        The encoded labels for the observation sequences used to fit the classifier.
    """

    def __init__(self, k, classes, weighting=None, window=1., use_c=False, independent=False, random_state=None):
        self._val = _Validator()
        self._k = self._val.is_restricted_integer(
            k, lambda x: x > 0, desc='number of neighbors', expected='greater than zero')
        self._window = float(window) if window in (0, 1) else self._val.is_restricted_float(
            window, lambda x: 0. <= x <= 1., desc='Sakoe-Chiba band width (fraction)', expected='between zero and one')
        self._random_state = self._val.is_random_state(random_state)

        self._val.is_iterable(classes, 'classes')
        self._val.is_string_or_numeric(classes[0], 'each class')
        if all(isinstance(label, type(classes[0])) for label in classes[1:]):
            self._encoder_ = LabelEncoder().fit(classes)
        else:
            raise TypeError('Expected all classes to be of the same type')

        self._weighting = self._val.is_func(weighting, 'distance weighting function') if weighting else lambda x: np.ones_like(x)

        self._use_c = self._val.is_boolean(use_c, desc='whether or not to use fast pure C compiled functions')
        if self._use_c and (dtw_cc is None):
            warnings.warn('DTAIDistance C library not available – using Python implementation', ImportWarning)
            self._use_c = False

        self._independent = self._val.is_boolean(independent, 'independent')
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
        X, y = self._val.is_observation_sequences_and_labels(X, y, dtype=np.float64)
        self._X_, self._y_ = X, self._encoder_.transform(y)
        self._n_features_ = X[0].shape[1]
        return self

    def predict(self, X, return_scores=False, original_labels=True, verbose=True, n_jobs=1):
        """Predicts the label for an observation sequence (or multiple sequences).

        Parameters
        ----------
        X: numpy.ndarray (float) or list of numpy.ndarray (float)
            An individual observation sequence or a list of multiple observation sequences.

        return_scores: bool
            Whether to return the scores for each class on the observation sequence(s).

        original_labels: bool
            Whether to inverse-transform the labels to their original encoding.

        verbose: bool
            Whether to display a progress bar or not.

            .. note::
                The progress bar cannot be displayed if both ``verbose=True`` and ``n_jobs > 1``.

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
        (self.X_, self.y_)
        X = self._val.is_observation_sequences(X, allow_single=True, dtype=np.float64)
        self._val.is_boolean(original_labels, desc='original_labels')
        self._val.is_boolean(verbose, desc='verbose')
        self._val.is_restricted_integer(n_jobs, lambda x: x == -1 or x > 0, 'number of jobs', '-1 or greater than zero')

        n_jobs = min(cpu_count() if n_jobs == -1 else n_jobs, len(X))

        # Make prediction for a single sequence
        if isinstance(X, np.ndarray):
            if n_jobs > 1:
                warnings.warn('Single predictions do not yet support multi-processing. Set n_jobs=1 to silence this warning.')

            output = self._predict(X, verbose=verbose)

        # Make predictions for multiple sequences (in parallel)
        else:
            if n_jobs == 1:
                # Use entire list as a chunk
                output = self._chunk_predict(X, verbose=verbose)

            else:
                if verbose:
                    warnings.warn('Progress bars cannot be displayed when using multiple processes. Set verbose=False to silence this warning.')

                # Split X into n_jobs equally sized chunks and process in parallel
                chunks = [list(chunk) for chunk in np.array_split(np.array(X, dtype=object), n_jobs)]
                output = np.vstack(Parallel(n_jobs=n_jobs)(delayed(self._chunk_predict)(chunk) for chunk in chunks))

        return self._output(output, return_scores=return_scores, original_labels=original_labels)

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
        X, y = self._val.is_observation_sequences_and_labels(X, y, dtype=np.float64)
        self._val.is_boolean(verbose, desc='verbose')
        predictions = self.predict(X, return_scores=False, original_labels=False, verbose=verbose, n_jobs=n_jobs)
        cm = confusion_matrix(self._encoder_.transform(y), predictions, labels=self._encoder_.transform(self._encoder_.classes_))
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
        (self.X_, self.y_)
        with open(path, 'wb') as file:
            pickle.dump({
                'k': self._k,
                'classes': self._encoder_.classes_,
                'weighting': marshal.dumps((self._weighting.__code__, self._weighting.__name__)),
                'window': self._window,
                'use_c': self._use_c,
                'independent': self._independent,
                'random_state': self._random_state,
                'X': self._X_,
                'y': self._y_,
                'n_features': self._n_features_
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
        clf._X_, clf._y_ = data['X'], data['y']
        clf._n_features_ = data['n_features']

        return clf

    def _dtw_1d(self, a, b, window):
        """Computes the DTW distance between two univariate sequences."""
        return dtw.distance(a, b, use_c=self._use_c, window=window)

    def _dtwi(self, A, B):
        """Computes the multivariate DTW distance as the sum of the pairwise per-feature DTW distances, allowing each feature to be warped independently."""
        window = max(1, int(self._window * max(len(A), len(B))))
        return np.sum([self._dtw_1d(A[:, i], B[:, i], window=window) for i in range(self._n_features_)])

    def _dtwd(self, A, B):
        """Computes the multivariate DTW distance so that the warping of the features depends on each other, by modifying the local distance measure."""
        window = max(1, int(self._window * max(len(A), len(B))))
        return dtw_ndim.distance(A, B, use_c=self._use_c, window=window)

    def _multi_argmax(self, arr):
        """Same as numpy.argmax but returns all occurrences of the maximum and only requires a single pass.
        From: https://stackoverflow.com/a/58652335
        """
        all_, max_ = [0], arr[0]
        for i in range(1, len(arr)):
            if arr[i] > max_:
                all_, max_ = [i], arr[i]
            elif arr[i] == max_:
                all_.append(i)
        return np.array(all_)

    def _find_k_nearest(self, distances):
        """Returns the labels and weightings (or scores) of the k-nearest neighbors"""
        idx = np.argpartition(distances, self._k)[:self._k]
        return self._y_[idx], self._weighting(distances[idx])

    def _find_max_labels(self, nearest_labels, nearest_scores):
        """Returns the mode label(s) of the k nearest neighbors.
        Vectorization from: https://stackoverflow.com/a/49239335
        """
        # Sort the labels in ascending order (and sort distances in the same order)
        sorted_labels_idx = nearest_labels.argsort()
        sorted_labels, sorted_scores = nearest_labels[sorted_labels_idx], nearest_scores[sorted_labels_idx]
        # Identify the indices where the sorted labels change (so we can group by labels)
        change_idx = np.concatenate(([0], np.nonzero(np.diff(sorted_labels))[0] + 1))
        # Calculate the total score for each label
        label_scores = np.add.reduceat(sorted_scores, change_idx)
        # Find the change index of the maximum score(s)
        max_score_idx = change_idx[self._multi_argmax(label_scores)]
        # Map the change index of the maximum scores back to the actual label(s)
        max_labels = sorted_labels[max_score_idx]
        # Store class scores
        scores = np.full(len(self.classes_), -np.inf)
        scores[sorted_labels[change_idx]] = label_scores
        # Map the change index of the maximum scores back to the actual label(s), and return scores
        return max_labels, scores

    def _predict(self, x1, verbose=False):
        """Makes a prediction for a single observation sequence."""
        # Calculate DTW distances between x1 and all other sequences
        distances = np.array([self._dtw(x1, x2) for x2 in tqdm(self._X_, desc='Calculating distances', disable=not(verbose))])
        # Find the k-nearest neighbors by DTW distance
        nearest_labels, nearest_scores = self._find_k_nearest(distances)
        # Out of the k-nearest neighbors, find the label(s) which had the highest total weighting
        max_labels, scores = self._find_max_labels(nearest_labels, nearest_scores)
        # Randomly pick from the set of labels with the maximum label score
        label = self._random_state.choice(max_labels, size=1)
        # Combine the label with the scores
        return np.concatenate((label, scores))

    def _chunk_predict(self, chunk, verbose=False):
        """Makes predictions for multiple observation sequences."""
        return np.array([self._predict(x, verbose=False) for x in tqdm(chunk, desc='Predicting', disable=not(verbose))])

    def _output(self, output, return_scores, original_labels):
        """Splits the label from the scores, inverse-transforms the labels if necessary, and returns the result."""
        if output.ndim == 1:
            labels, scores = int(output[0]), output[1:]
            labels = self._encoder_.inverse_transform([labels]).item() if original_labels else labels
        else:
            labels, scores = output[:, 0].astype(int), output[:, 1:]
            labels = self._encoder_.inverse_transform(labels) if original_labels else labels
        return (labels, scores) if return_scores else labels

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
    def encoder_(self):
        return self._encoder_

    @property
    def classes_(self):
        return self._encoder_.classes_

    @property
    def X_(self):
        try:
            return self._X_
        except AttributeError:
            raise RuntimeError('The classifier needs to be fitted first')

    @property
    def y_(self):
        try:
            return self._y_
        except AttributeError:
            raise RuntimeError('The classifier needs to be fitted first')

    def __repr__(self):
        name = '.'.join([self.__class__.__module__.split('.')[0], self.__class__.__name__])
        attrs = [
            ('k', repr(self._k)),
            ('window', repr(self._window)),
            ('use_c', repr(self._use_c)),
            ('independent', repr(self._independent)),
            ('classes', repr(list(self._encoder_.classes_)))
        ]
        return '{}({})'.format(name, ', '.join('{}={}'.format(name, val) for name, val in attrs))