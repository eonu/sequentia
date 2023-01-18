from __future__ import annotations

from types import SimpleNamespace
from typing import Optional, Union, Callable
from joblib import Parallel, delayed

import numpy as np
from pydantic import NegativeInt, NonNegativeInt, PositiveInt, confloat
from numba import njit, prange
from sklearn.utils import check_random_state

from sequentia.models.knn.base import _KNNMixin, _KNNValidator
from sequentia.models.base import _Classifier

from sequentia.utils.decorators import _validate_params, _requires_fit, _override_params
from sequentia.utils.data import SequentialDataset
from sequentia.utils.multiprocessing import _effective_n_jobs
from sequentia.utils.validation import (
    _check_classes,
    Array,
    _MultivariateFloatSequenceClassifierValidator
)

__all__ = ['KNNClassifier']

_defaults = SimpleNamespace(
    **{
        **_KNNMixin._defaults.__dict__,
        "classes": None,
    }
)


class _KNNClassifierValidator(_KNNValidator):
    classes: Optional[Array[int]] = _defaults.classes


class KNNClassifier(_KNNMixin, _Classifier):
    """A k-nearest neighbor classifier that uses DTW as a distance measure for sequence comparison.

    The classifier computes the score for each class as the total of the distance weightings of every sequence belonging to that class,
    within the DTW k-neighborhood of the sequence being classified.

    Examples
    --------
    Using a :class:`.KNNClassifier` to classify spoken digits. ::

        import numpy as np
        from sequentia.datasets import load_digits
        from sequentia.models.knn import KNNClassifier

        # Seed for reproducible pseudo-randomness
        random_state = np.random.RandomState(1)

        # Fetch MFCCs of spoken digits
        data = load_digits()
        train_data, test_data = data.split(test_size=0.2, random_state=random_state)

        # Create a HMMClassifier using a class frequency prior
        clf = KNNClassifier()

        # Fit the classifier
        X_train, y_train, lengths_train = train_data.X_y_lengths
        clf.fit(X_train, y_train, lengths_train)

        # Predict classes for the test observation sequences
        X_test, lengths_test = test_data.X_lengths
        y_pred = clf.predict(X_test, lengths_test)
    """

    _defaults = _defaults


    @_validate_params(using=_KNNClassifierValidator)
    def __init__(
        self,
        *,
        k: PositiveInt = _defaults.k,
        weighting: Optional[Callable] = _defaults.weighting,
        window: confloat(ge=0, le=1) = _defaults.window,
        independent: bool = _defaults.independent,
        classes: Optional[Array[int]] = None,
        use_c: bool = _defaults.use_c,
        n_jobs: Union[NegativeInt, PositiveInt] = _defaults.n_jobs,
        random_state: Optional[Union[NonNegativeInt, np.random.RandomState]] = _defaults.random_state
    ) -> KNNClassifier:
        """Initializes the :class:`.KNNClassifier`.

        :param k: Number of neighbors.

        :param weighting: A callable that specifies how distance weighting should be performed.
            The callable should accept a :class:`numpy:numpy.ndarray` of DTW distances, apply an element-wise weighting transformation
            to the matrix of DTW distances, then return an equally-sized :class:`numpy:numpy.ndarray` of weightings.
            If ``None``, then a uniform weighting of 1 will be applied to all distances.

        :param window: The size of the Sakoe—Chiba band global constrant as a fraction of the length of the shortest of the two sequences being compared.

            - A larger window will give more freedom to the DTW alignment, allowing more deviation but leading to potentially slower computation.
              A window of 1 is equivalent to full DTW computation with no global constraint applied.
            - A smaller window will restrict the DTW alignment, and possibly speed up the DTW computation.
              A window of 0 is equivalent to Euclidean distance.

        :param independent: Whether or not to allow features to be warped independently from each other. See [#dtw_multi]_ for an overview of independent and dependent dynamic time warping.

        :param classes: Set of possible class labels.

            - If not provided, these will be determined from the training data labels.
            - If provided, output from methods such as :func:`predict_proba` and :func:`predict_scores`
              will follow the ordering of the class labels provided here.

        :param use_c: Whether or not to use fast pure C compiled functions from `dtaidistance <https://github.com/wannesm/dtaidistance>`__ to perform the DTW computations.

        :param n_jobs: Maximum number of concurrently running workers.

            - If 1, no parallelism is used at all (useful for debugging).
            - If -1, all CPUs are used.
            - If < -1, ``(n_cpus + 1 + n_jobs)`` are used — e.g. ``n_jobs=-2`` uses all but one.

        :param random_state: Seed or :class:`numpy:numpy.random.RandomState` object for reproducible pseudo-randomness.
        """
        #: Number of neighbors.
        self.k = k
        #: A callable that specifies how distance weighting should be performed.
        self.weighting = weighting
        #: The size of the Sakoe—Chiba band global constrant as a fraction of the length of the shortest of the two sequences being compared.
        self.window = window
        #: Whether or not to allow features to be warped independently from each other.
        self.independent = independent
        #: Set of possible class labels.
        self.classes = classes
        #: Whether or not to use fast pure C compiled functions from `dtaidistance <https://github.com/wannesm/dtaidistance>`__ to perform the DTW computations.
        self.use_c = use_c
        #: Maximum number of concurrently running workers.
        self.n_jobs = n_jobs
        #: Seed or :class:`numpy:numpy.random.RandomState` object for reproducible pseudo-randomness.
        self.random_state = random_state


    def fit(
        self,
        X: Array[float],
        y: Array[int],
        lengths: Optional[Array[int]] = None
    ) -> KNNClassifier:
        """Fits the classifier to the sequence(s) in ``X``.

        :param X: Univariate or multivariate observation sequence(s).

            - Should be a single 1D or 2D array.
            - Should have length as the 1st dimension and features as the 2nd dimension.
            - Should be a concatenated sequence if multiple sequences are provided,
              with respective sequence lengths being provided in the ``lengths`` argument for decoding the original sequences.

        :param y: Classes corresponding to sequence(s) provided in ``X``.

        :param lengths: Lengths of the observation sequence(s) provided in ``X``.

            - If ``None``, then ``X`` is assumed to be a single observation sequence.
            - ``len(X)`` should be equal to ``sum(lengths)``.

        :return: The fitted classifier.
        """
        data = _MultivariateFloatSequenceClassifierValidator(X=X, y=y, lengths=lengths)
        self.X_ = data.X
        self.y_ = data.y
        self.lengths_ = data.lengths
        self.idxs_ = SequentialDataset._get_idxs(data.lengths)
        self.random_state_ = check_random_state(self.random_state)
        self.classes_ = _check_classes(data.y, self.classes)
        return self


    @_requires_fit
    def predict(
        self,
        X: Array[float],
        lengths: Optional[Array[int]] = None
    ) -> Array[int]:
        """Predicts classes for the sequence(s) in ``X``.

        :param X: Univariate or multivariate observation sequence(s).

            - Should be a single 1D or 2D array.
            - Should have length as the 1st dimension and features as the 2nd dimension.
            - Should be a concatenated sequence if multiple sequences are provided,
              with respective sequence lengths being provided in the ``lengths`` argument for decoding the original sequences.

        :param lengths: Lengths of the observation sequence(s) provided in ``X``.

            - If ``None``, then ``X`` is assumed to be a single observation sequence.
            - ``len(X)`` should be equal to ``sum(lengths)``.

        :note: This method requires a trained classifier — see :func:`fit`.

        :return: Class predictions.
        """
        class_scores = self.predict_scores(X, lengths)
        return self._find_max_labels(class_scores)


    def fit_predict(
        self,
        X: Array[float],
        y: Array[int],
        lengths: Optional[Array[int]] = None
    ) -> Array[int]:
        """Fits the classifier to the sequence(s) in ``X`` and predicts classes for ``X``.

        :param X: Univariate or multivariate observation sequence(s).

            - Should be a single 1D or 2D array.
            - Should have length as the 1st dimension and features as the 2nd dimension.
            - Should be a concatenated sequence if multiple sequences are provided,
              with respective sequence lengths being provided in the ``lengths`` argument for decoding the original sequences.

        :param y: Classes corresponding to sequence(s) provided in ``X``.

        :param lengths: Lengths of the observation sequence(s) provided in ``X``.

            - If ``None``, then ``X`` is assumed to be a single observation sequence.
            - ``len(X)`` should be equal to ``sum(lengths)``.

        :return: Class predictions.
        """
        return super().fit_predict(X, y, lengths)


    @_requires_fit
    def predict_proba(
        self,
        X: Array[float],
        lengths: Optional[Array[int]] = None
    ) -> Array[float]:
        """Predicts class probabilities for the sequence(s) in ``X``.

        Probabilities are calculated as normalized class scores.

        :param X: Univariate or multivariate observation sequence(s).

            - Should be a single 1D or 2D array.
            - Should have length as the 1st dimension and features as the 2nd dimension.
            - Should be a concatenated sequence if multiple sequences are provided,
              with respective sequence lengths being provided in the ``lengths`` argument for decoding the original sequences.

        :param lengths: Lengths of the observation sequence(s) provided in ``X``.

            - If ``None``, then ``X`` is assumed to be a single observation sequence.
            - ``len(X)`` should be equal to ``sum(lengths)``.

        :note: This method requires a trained classifier — see :func:`fit`.

        :return: Class membership probabilities.
        """
        class_scores = self.predict_scores(X, lengths)
        return class_scores / class_scores.sum(axis=1, keepdims=True)


    @_requires_fit
    def predict_scores(
        self,
        X: Array[float],
        lengths: Optional[Array[int]] = None
    ) -> Array[float]:
        """Predicts class scores for the sequence(s) in ``X``.

        Scores are calculated as the class distance weighting sums of all training sequences in the k-neighborhood.

        :param X: Univariate or multivariate observation sequence(s).

            - Should be a single 1D or 2D array.
            - Should have length as the 1st dimension and features as the 2nd dimension.
            - Should be a concatenated sequence if multiple sequences are provided,
              with respective sequence lengths being provided in the ``lengths`` argument for decoding the original sequences.

        :param lengths: Lengths of the observation sequence(s) provided in ``X``.

            - If ``None``, then ``X`` is assumed to be a single observation sequence.
            - ``len(X)`` should be equal to ``sum(lengths)``.

        :note: This method requires a trained classifier — see :func:`fit`.

        :return: Class scores.
        """
        _, k_distances, k_labels = self.query_neighbors(X, lengths, sort=False)
        k_weightings = self._weighting()(k_distances)
        return self._compute_scores(k_labels, k_weightings)


    @_requires_fit
    def score(
        self,
        X: Array,
        y: Array[int],
        lengths: Optional[Array[int]],
        normalize: bool = True,
        sample_weight: Optional[Array] = None,
    ) -> float:
        """Calculates accuracy for the sequence(s) in ``X``.

        :param X: Univariate or multivariate observation sequence(s).

            - Should be a single 1D or 2D array.
            - Should have length as the 1st dimension and features as the 2nd dimension.
            - Should be a concatenated sequence if multiple sequences are provided,
              with respective sequence lengths being provided in the ``lengths`` argument for decoding the original sequences.

        :param y: Classes corresponding to the observation sequence(s) in ``X``.

        :param lengths: Lengths of the observation sequence(s) provided in ``X``.

            - If ``None``, then ``X`` is assumed to be a single observation sequence.
            - ``len(X)`` should be equal to ``sum(lengths)``.

        :param normalize: See :func:`sklearn:sklearn.metrics.accuracy_score`.

        :param sample_weight: See :func:`sklearn:sklearn.metrics.accuracy_score`.

        :note: This method requires a trained classifier — see :func:`fit`.

        :return: Classification accuracy.
        """
        return super().score(X, y, lengths, normalize, sample_weight)


    @_validate_params(using=_KNNValidator)
    @_override_params(_KNNValidator.fields(), temporary=False)
    def set_params(self, **kwargs):
        return self


    def _compute_scores(
        self,
        labels: Array[int],
        weightings: Array[float]
    ) -> Array[float]:
        """Calculates the sum of the weightings for each label group.

        TODO
        """
        scores = np.zeros((len(labels), len(self.classes_)))
        for i, k in enumerate(self.classes_):
            scores[:, i] = np.einsum('ij,ij->i', labels == k, weightings)
        return scores


    def _find_max_labels(
        self,
        scores: Array[float]
    ) -> Array[int]:
        """Returns the label of the k nearest neighbors with the highest score for each example.

        TODO
        """
        n_jobs = _effective_n_jobs(self.n_jobs, scores)
        score_chunks = np.array_split(scores, n_jobs)
        return np.concatenate(
            Parallel(n_jobs=n_jobs, max_nbytes=None)(
                delayed(self._find_max_labels_chunk)(score_chunk)
                for score_chunk in score_chunks
            )
        )


    def _find_max_labels_chunk(
        self,
        score_chunk: Array[float]
    ) -> Array[int]:
        """Returns the label with the highest score for each item in the chunk.

        TODO
        """
        max_labels = np.zeros(len(score_chunk), dtype=int)
        for i, scores in enumerate(score_chunk):
            max_score_idxs = self._multi_argmax(scores)
            max_labels[i] = self.random_state_.choice(self.classes_[max_score_idxs], size=1)
        return max_labels


    @staticmethod
    @njit
    def _multi_argmax(
        arr: Array
    ) -> Array[int]:
        """Same as numpy.argmax but returns all occurrences of the maximum and only requires a single pass.
        From: https://stackoverflow.com/a/58652335

        TODO
        """
        all_, max_ = [0], arr[0]
        for i in prange(1, len(arr)):
            if arr[i] > max_:
                all_, max_ = [i], arr[i]
            elif arr[i] == max_:
                all_.append(i)
        return np.array(all_)
