# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""A k-nearest neighbor classifier that uses DTW as a distance measure for
sequence comparison.
"""

from __future__ import annotations

import typing as t

import joblib
import numba
import numpy as np
import pydantic as pyd

from sequentia._internal import _data, _multiprocessing, _sklearn, _validation
from sequentia._internal._typing import Array, FloatArray, IntArray
from sequentia.models.base import ClassifierMixin
from sequentia.models.knn.base import KNNMixin

__all__ = ["KNNClassifier"]


class KNNClassifier(KNNMixin, ClassifierMixin):
    """A k-nearest neighbor classifier that uses DTW as a distance measure for
    sequence comparison.

    The classifier computes the score for each class as the total of the
    distance weightings of every sequence belonging to that class,
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
        clf.fit(train_data.X, train_data.y, lengths=train_data.lengths)

        # Predict classes for the test observation sequences
        y_pred = clf.predict(test_data.X, lengths=test_data.lengths)
    """  # noqa: E501

    @pyd.validate_call(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self: pyd.SkipValidation,
        *,
        k: pyd.PositiveInt = 1,
        weighting: t.Callable[[FloatArray], FloatArray] | None = None,
        window: pyd.confloat(ge=0.0, le=1.0) = 1.0,
        independent: bool = False,
        use_c: bool = False,
        n_jobs: pyd.PositiveInt | pyd.NegativeInt = 1,
        random_state: pyd.NonNegativeInt | np.random.RandomState | None = None,
        classes: list[int] | None = None,
    ) -> pyd.SkipValidation:
        """Initializes the :class:`.KNNClassifier`.

        Parameters
        ----------
        self: KNNClassifier

        k:
            Number of neighbors.

        weighting:
            A callable that specifies how distance weighting should be
            performed.

            The callable should accept a :class:`numpy:numpy.ndarray` of DTW
            distances, apply an element-wise weighting transformation to the
            matrix of DTW distances, then return an equally-sized
            :class:`numpy:numpy.ndarray` of weightings.

            If ``None``, then a uniform weighting of 1 will be applied to all
            distances.

        window:
            The size of the Sakoe—Chiba band global constrant as a fraction
            of the length of the shortest of the two sequences being compared.

            - A larger window will give more freedom to the DTW alignment,
              allowing more deviation but leading to potentially slower
              computation.
              A window of 1 is equivalent to full DTW computation with no
              global constraint applied.
            - A smaller window will restrict the DTW alignment, and possibly
              speed up the DTW computation.
              A window of 0 is equivalent to Euclidean distance.

        independent:
            Whether or not to allow features to be warped independently from
            each other. See [#dtw_multi]_ for an overview of independent and
            dependent dynamic time warping.

        use_c:
            Whether or not to use fast pure C compiled functions from
            `dtaidistance <https://github.com/wannesm/dtaidistance>`__ to
            perform the DTW computations.

        n_jobs:
            Maximum number of concurrently running workers.

            - If 1, no parallelism is used at all (useful for debugging).
            - If -1, all CPUs are used.
            - If < -1, ``(n_cpus + 1 + n_jobs)`` are used — e.g. ``n_jobs=-2``
              uses all but one.

        random_state:
            Seed or :class:`numpy:numpy.random.RandomState` object for
            reproducible pseudo-randomness.

        classes:
            Set of possible class labels.

            - If not provided, these will be determined from the training data
              labels.
            - If provided, output from methods such as :func:`predict_proba`
              and :func:`predict_scores` will follow the ordering of the class
              labels provided here.

        Returns
        -------
        KNNClassifier
        """
        self.k: int = k
        """Number of neighbors."""

        self.weighting: t.Callable[[np.ndarray], np.ndarray] | None = (
            weighting  # placeholder
        )
        """A callable that specifies how distance weighting should be
        performed."""

        self.window: float = window
        """The size of the Sakoe—Chiba band global constrant as a fraction of
        the length of the shortest of the two sequences being compared."""

        self.independent: bool = independent
        """Whether or not to allow features to be warped independently from
        each other."""

        self.use_c: bool = use_c
        """Whether or not to use fast pure C compiled functions from
        `dtaidistance <https://github.com/wannesm/dtaidistance>`__ to
        perform the DTW computations."""

        self.n_jobs: int = n_jobs
        """Maximum number of concurrently running workers."""

        self.random_state: int | np.random.RandomState | None = random_state
        """Seed or :class:`numpy:numpy.random.RandomState` object for
        reproducible pseudo-randomness."""

        self.classes: list[int] | None = classes
        """Set of possible class labels."""

        # Allow metadata routing for lengths
        if _sklearn.routing_enabled():
            self.set_fit_request(lengths=True)
            self.set_predict_request(lengths=True)
            self.set_predict_log_proba_request(lengths=True)
            self.set_predict_proba_request(lengths=True)
            self.set_score_request(
                lengths=True,
                normalize=True,
                sample_weight=True,
            )

    def fit(
        self: KNNClassifier,
        X: FloatArray,
        y: IntArray,
        *,
        lengths: IntArray | None = None,
    ) -> KNNClassifier:
        """Fit the classifier to the sequence(s) in ``X``.

        Parameters
        ----------
        self: KNNClassifier

        X:
            Sequence(s).

        y:
            Classes corresponding to sequence(s) in ``X``.

        lengths:
            Lengths of the sequence(s) provided in ``X``.

            - If ``None``, then ``X`` is assumed to be a single sequence.
            - ``len(X)`` should be equal to ``sum(lengths)``.

        Returns
        -------
        KNNClassifier:
            The fitted classifier.
        """
        self.X_, self.lengths_ = _validation.check_X_lengths(
            X, lengths=lengths, dtype=self._DTYPE
        )
        self.y_ = _validation.check_y(
            y,
            lengths=self.lengths_,
            dtype=np.int8,
        )
        self.idxs_ = _data.get_idxs(self.lengths_)
        self.use_c_ = _validation.check_use_c(self.use_c)
        self.random_state_ = _validation.check_random_state(self.random_state)
        self.classes_ = _validation.check_classes(
            self.y_,
            classes=self.classes,
        )
        _validation.check_weighting(self.weighting)
        return self

    @_validation.requires_fit
    def predict(
        self: KNNClassifier,
        X: FloatArray,
        *,
        lengths: IntArray | None = None,
    ) -> IntArray:
        """Predict classes for the sequence(s) in ``X``.

        Parameters
        ----------
        self: KNNClassifier

        X:
            Sequence(s).

        lengths:
            Lengths of the sequence(s) provided in ``X``.

            - If ``None``, then ``X`` is assumed to be a single sequence.
            - ``len(X)`` should be equal to ``sum(lengths)``.

        Returns
        -------
        numpy.ndarray:
            Class predictions.

        Notes
        -----
        This method requires a trained classifier — see :func:`fit`.
        """
        class_scores = self.predict_scores(X, lengths=lengths)
        return self._find_max_labels(class_scores)

    @_validation.requires_fit
    def predict_log_proba(
        self: KNNClassifier,
        X: FloatArray,
        *,
        lengths: IntArray | None = None,
    ) -> FloatArray:
        """Predict log class probabilities for the sequence(s) in ``X``.

        Probabilities are calculated as normalized class scores.

        Parameters
        ----------
        self: KNNClassifier

        X:
            Sequence(s).

        lengths:
            Lengths of the sequence(s) provided in ``X``.

            - If ``None``, then ``X`` is assumed to be a single sequence.
            - ``len(X)`` should be equal to ``sum(lengths)``.

        Returns
        -------
        numpy.ndarray:
            Class membership log-probabilities.

        Notes
        -----
        This method requires a trained classifier — see :func:`fit`.
        """
        return np.log(self.predict_proba(X, lengths=lengths))

    @_validation.requires_fit
    def predict_proba(
        self: KNNClassifier,
        X: FloatArray,
        *,
        lengths: IntArray | None = None,
    ) -> FloatArray:
        """Predict class probabilities for the sequence(s) in ``X``.

        Probabilities are calculated as normalized class scores.

        Parameters
        ----------
        self: KNNClassifier

        X:
            Sequence(s).

        lengths:
            Lengths of the sequence(s) provided in ``X``.

            - If ``None``, then ``X`` is assumed to be a single sequence.
            - ``len(X)`` should be equal to ``sum(lengths)``.

        Returns
        -------
        numpy.ndarray:
            Class membership probabilities.

        Notes
        -----
        This method requires a trained classifier — see :func:`fit`.
        """
        class_scores = self.predict_scores(X, lengths=lengths)
        return class_scores / class_scores.sum(axis=1, keepdims=True)

    @_validation.requires_fit
    def predict_scores(
        self: KNNClassifier,
        X: FloatArray,
        *,
        lengths: IntArray | None = None,
    ) -> FloatArray:
        """Predict class scores for the sequence(s) in ``X``.

        Scores are calculated as the class distance weighting sums of all
        training sequences in the k-neighborhood.

        Parameters
        ----------
        self: KNNClassifier

        X:
            Sequence(s).

        lengths:
            Lengths of the sequence(s) provided in ``X``.

            - If ``None``, then ``X`` is assumed to be a single sequence.
            - ``len(X)`` should be equal to ``sum(lengths)``.

        Returns
        -------
        numpy.ndarray:
            Class scores.

        Notes
        -----
        This method requires a trained classifier — see :func:`fit`.
        """
        _, k_distances, k_labels = self.query_neighbors(
            X,
            lengths=lengths,
            sort=False,
        )
        k_weightings = self._weighting()(k_distances)
        return self._compute_scores(k_labels, k_weightings)

    def _compute_scores(
        self: KNNClassifier, labels: IntArray, weightings: FloatArray
    ) -> FloatArray:
        """Calculate the sum of the weightings for each label group."""
        scores = np.zeros((len(labels), len(self.classes_)))
        for i, k in enumerate(self.classes_):
            scores[:, i] = np.einsum("ij,ij->i", labels == k, weightings)
        return scores

    def _find_max_labels(
        self: KNNClassifier,
        scores: FloatArray,
        /,
    ) -> IntArray:
        """Return the label of the k nearest neighbors with the highest score
        for each example.
        """
        n_jobs = _multiprocessing.effective_n_jobs(self.n_jobs, x=scores)
        score_chunks = np.array_split(scores, n_jobs)
        return np.concatenate(
            joblib.Parallel(n_jobs=n_jobs, max_nbytes=None)(
                joblib.delayed(self._find_max_labels_chunk)(score_chunk)
                for score_chunk in score_chunks
            )
        )

    def _find_max_labels_chunk(
        self: KNNClassifier, score_chunk: FloatArray, /
    ) -> IntArray:
        """Return the label with the highest score for each item in the
        chunk.
        """
        max_labels = np.zeros(len(score_chunk), dtype=int)
        for i, scores in enumerate(score_chunk):
            max_score_idxs = self._multi_argmax(scores)
            max_labels[i] = self.random_state_.choice(
                self.classes_[max_score_idxs], size=1
            ).item()
        return max_labels

    @staticmethod
    @numba.njit
    def _multi_argmax(arr: Array, /) -> IntArray:
        """Same as numpy.argmax but returns all occurrences of the maximum
        and only requires a single pass.

        From: https://stackoverflow.com/a/58652335
        """
        all_, max_ = [0], arr[0]
        for i in numba.prange(1, len(arr)):
            if arr[i] > max_:
                all_, max_ = [i], arr[i]
            elif arr[i] == max_:
                all_.append(i)
        return np.array(all_)
