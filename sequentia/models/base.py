# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Base classifier and regressor mixin classes."""

from __future__ import annotations

import abc

import numpy as np
import sklearn.base
import sklearn.metrics

from sequentia._internal import _validation
from sequentia._internal._typing import Array, FloatArray, IntArray

__all__ = ["ClassifierMixin", "RegressorMixin"]


class ClassifierMixin(
    sklearn.base.BaseEstimator,
    sklearn.base.ClassifierMixin,
    metaclass=abc.ABCMeta,
):
    """Represents a generic sequential classifier."""

    @abc.abstractmethod
    def fit(
        self: ClassifierMixin,
        X: Array,
        y: IntArray,
        *,
        lengths: IntArray | None = None,
    ) -> ClassifierMixin:
        """Fit the classifier with the provided sequences and outputs."""
        raise NotImplementedError

    @abc.abstractmethod
    def predict(
        self: ClassifierMixin,
        X: Array,
        *,
        lengths: IntArray | None = None,
    ) -> IntArray:
        """Predict outputs for the provided sequences."""
        raise NotImplementedError

    def fit_predict(
        self: ClassifierMixin,
        X: Array,
        y: IntArray,
        *,
        lengths: IntArray | None = None,
    ) -> IntArray:
        """Fit the model to the sequence(s) in ``X`` and predicts outputs for
        ``X``.

        Parameters
        ----------
        self: ClassifierMixin

        X:
            Sequence(s).

        y:
            Outputs corresponding to sequence(s) in ``X``.

        lengths:
            Lengths of the sequence(s) provided in ``X``.

            - If ``None``, then ``X`` is assumed to be a single sequence.
            - ``len(X)`` should be equal to ``sum(lengths)``.

        Returns
        -------
        numpy.ndarray:
            Output predictions.
        """
        return self.fit(X, y, lengths=lengths).predict(X, lengths=lengths)

    @abc.abstractmethod
    def predict_proba(
        self: ClassifierMixin,
        X: Array,
        *,
        lengths: IntArray | None = None,
    ) -> FloatArray:
        """Predict class probabilities for the provided sequences."""
        raise NotImplementedError

    @abc.abstractmethod
    def predict_scores(
        self: ClassifierMixin,
        X: Array,
        *,
        lengths: IntArray | None = None,
    ) -> FloatArray:
        """Predict class scores for the provided sequences."""
        raise NotImplementedError

    @_validation.requires_fit
    def score(
        self: ClassifierMixin,
        X: Array,
        y: IntArray,
        *,
        lengths: IntArray | None = None,
        normalize: bool = True,
        sample_weight: Array | None = None,
    ) -> float:
        """Calculate the predictive accuracy for the sequence(s) in ``X``.

        Parameters
        ----------
        self: ClassifierMixin

        X:
            Sequence(s).

        y:
            Outputs corresponding to sequence(s) in ``X``.

        lengths:
            Lengths of the sequence(s) provided in ``X``.

            - If ``None``, then ``X`` is assumed to be a single sequence.
            - ``len(X)`` should be equal to ``sum(lengths)``.

        normalize:
            See :func:`sklearn:sklearn.metrics.accuracy_score`.

        sample_weight:
            See :func:`sklearn:sklearn.metrics.accuracy_score`.

        Returns
        -------
        float
            Predictive accuracy.

        Notes
        -----
        This method requires a trained classifier — see :func:`fit`.
        """
        y = _validation.check_y(y, lengths=lengths, dtype=np.int8)
        y_pred = self.predict(X, lengths=lengths)
        return sklearn.metrics.accuracy_score(
            y, y_pred, normalize=normalize, sample_weight=sample_weight
        )


class RegressorMixin(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    """Represents a generic sequential regressor."""

    @abc.abstractmethod
    def fit(
        self: RegressorMixin,
        X: FloatArray,
        y: FloatArray,
        *,
        lengths: IntArray | None = None,
    ) -> RegressorMixin:
        """Fit the regressor with the provided sequences and outputs."""
        raise NotImplementedError

    @abc.abstractmethod
    def predict(
        self: RegressorMixin, X: FloatArray, lengths: IntArray | None = None
    ) -> FloatArray:
        """Predict outputs for the provided sequences."""
        raise NotImplementedError

    def fit_predict(
        self: RegressorMixin,
        X: FloatArray,
        y: FloatArray,
        *,
        lengths: IntArray | None = None,
    ) -> FloatArray:
        """Fit the model to the sequence(s) in ``X`` and predicts outputs for
        ``X``.

        Parameters
        ----------
        self: RegressorMixin

        X:
            Sequence(s).

        y:
            Outputs corresponding to sequence(s) in ``X``.

        lengths:
            Lengths of the sequence(s) provided in ``X``.

            - If ``None``, then ``X`` is assumed to be a single sequence.
            - ``len(X)`` should be equal to ``sum(lengths)``.

        Returns
        -------
        numpy.ndarray
            Output predictions.
        """
        return self.fit(X, y, lengths=lengths).predict(X, lengths=lengths)

    @_validation.requires_fit
    def score(
        self: RegressorMixin,
        X: FloatArray,
        y: FloatArray,
        *,
        lengths: IntArray | None = None,
        sample_weight: Array | None = None,
    ) -> float:
        r"""Calculate the predictive coefficient of determination
        (R\ :sup:`2`) for the sequence(s) in ``X``.

        Parameters
        ----------
        self: RegressorMixin

        X:
            Sequence(s).

        y:
            Outputs corresponding to sequence(s) in ``X``.

        lengths:
            Lengths of the sequence(s) provided in ``X``.

            - If ``None``, then ``X`` is assumed to be a single sequence.
            - ``len(X)`` should be equal to ``sum(lengths)``.

        sample_weight:
            See :func:`sklearn:sklearn.metrics.r2_score`.

        Returns
        -------
        float
            Coefficient of determination.

        Notes
        -----
        This method requires a trained classifier — see :func:`fit`.
        """
        y = _validation.check_y(y, lengths=lengths, dtype=np.float64)
        y_pred = self.predict(X, lengths=lengths)
        return sklearn.metrics.r2_score(y, y_pred, sample_weight=sample_weight)
