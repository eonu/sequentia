from __future__ import annotations
from typing import Union, Optional, Callable, Literal
from joblib import Parallel, delayed

import numpy as np
from numba import njit, prange
from sklearn.utils import check_random_state

from sequentia.models.knn.base import KNNValidator, KNNMixin
from sequentia.models.base import Classifier

from sequentia.utils.decorators import validate_params, requires_fit, override_params
from sequentia.utils.data import SequentialDataset
from sequentia.utils.multiprocessing import effective_n_jobs
from sequentia.utils.validation import (
    check_classes,
    check_is_fitted,
    Array,
    MultivariateFloatSequenceClassifierValidator
)

__all__ = ['KNNClassifier']

class KNNClassifier(KNNMixin, Classifier):
    @validate_params(using=KNNValidator)
    def __init__(self, *,
        k: int = 1,
        weighting: Union[Literal['uniform'], Callable] = 'uniform', # TODO: Must be a non-negative matrix function!
        window: float = 1,
        independent: bool = False,
        classes: Optional[Array[int]] = None,
        use_c: bool = False,
        n_jobs: int = 1,
        random_state: Optional[Union[int, np.random.RandomState]] = None
    ):
        self.k = k
        self.weighting = weighting
        self.window = window
        self.independent = independent
        self.classes = classes
        self.use_c = use_c
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(
        self,
        X: Array[float],
        y: Array[int],
        lengths: Optional[Array[int]] = None
    ) -> KNNClassifier:
        data = MultivariateFloatSequenceClassifierValidator(X=X, y=y, lengths=lengths)
        self.X_ = data.X
        self.y_ = data.y
        self.lengths_ = data.lengths
        self.idxs_ = SequentialDataset._get_idxs(data.lengths)
        self.random_state_ = check_random_state(self.random_state)
        self.classes_ = check_classes(y, self.classes)
        return self

    @requires_fit
    def predict(
        self,
        X: Array[float],
        lengths: Optional[Array[int]] = None
    ) -> Array[int]:
        class_scores = self.predict_scores(X, lengths)
        return self._find_max_labels(class_scores)

    @requires_fit
    def predict_proba(
        self,
        X: Array[float],
        lengths: Optional[Array[int]] = None
    ) -> Array[float]:
        class_scores = self.predict_scores(X, lengths)
        return class_scores / class_scores.sum(axis=1, keepdims=True)

    @requires_fit
    def predict_scores(
        self,
        X: Array[float],
        lengths: Optional[Array[int]] = None
    ) -> Array[float]:
        _, k_distances, k_labels = self.query_neighbors(X, lengths, sort=False)
        k_weightings = self._weighting()(k_distances)
        return self._compute_scores(k_labels, k_weightings)

    @validate_params(using=KNNValidator)
    @override_params(KNNValidator.fields(), temporary=False)
    def set_params(self, **kwargs):
        return self

    def _compute_scores(
        self,
        labels: Array[int],
        weightings: Array[float]
    ) -> Array[float]:
        """Calculates the sum of the weightings for each label group."""
        scores = np.zeros((len(labels), len(self.classes_)))
        for i, k in enumerate(self.classes_):
            scores[:, i] = np.einsum('ij,ij->i', labels == k, weightings)
        return scores

    def _find_max_labels(
        self,
        scores: Array[float]
    ) -> Array[int]:
        """Returns the label of the k nearest neighbors with the highest score for each example."""
        n_jobs = effective_n_jobs(self.n_jobs, scores)
        score_chunks = np.array_split(scores, n_jobs)
        return np.concatenate(
            Parallel(n_jobs=n_jobs)(
                delayed(self._find_max_labels_chunk)(score_chunk)
                for score_chunk in score_chunks
            )
        )

    def _find_max_labels_chunk(
        self,
        score_chunk: Array[float]
    ) -> Array[int]:
        """Returns the label with the highest score for each item in the chunk."""
        max_labels = np.zeros(len(score_chunk), dtype=int)
        for i, scores in enumerate(score_chunk):
            max_score_idxs = self._multi_argmax(scores)
            max_labels[i] = self.random_state_.choice(self.classes_[max_score_idxs], size=1)
        return max_labels

    @staticmethod
    @njit
    def _multi_argmax(
        arr: Union[Array[int], Array[float]]
    ) -> Array[int]:
        """Same as numpy.argmax but returns all occurrences of the maximum and only requires a single pass.
        From: https://stackoverflow.com/a/58652335
        """
        all_, max_ = [0], arr[0]
        for i in prange(1, len(arr)):
            if arr[i] > max_:
                all_, max_ = [i], arr[i]
            elif arr[i] == max_:
                all_.append(i)
        return np.array(all_)

    def __eq__(self, other):
        eq = super().__eq__(other)
        self_fitted = check_is_fitted(self, ['classes_'], True)
        other_fitted = check_is_fitted(self, ['classes_'], True)
        eq &= self_fitted == other_fitted
        if self_fitted and other_fitted:
            eq &= np.array_equal(self.classes_, other.classes_)
        return eq
