# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""A k-nearest neighbor regressor that uses DTW as a distance measure for
sequence comparison.
"""

from __future__ import annotations

import typing as t

import numpy as np
import pydantic as pyd

from sequentia._internal import _data, _sklearn, _validation
from sequentia._internal._typing import FloatArray, IntArray
from sequentia.models.base import RegressorMixin
from sequentia.models.knn.base import KNNMixin

__all__ = ["KNNRegressor"]


class KNNRegressor(KNNMixin, RegressorMixin):
    """A k-nearest neighbor regressor that uses DTW as a distance measure for
    sequence comparison.

    The regressor computes the output as a distance weighted average of the
    outputs of the sequences within the DTW k-neighborhood of the sequence
    being predicted.
    """

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
    ) -> pyd.SkipValidation:
        """Initializes the :class:`.KNNRegressor`.

        Parameters
        ----------
        self: KNNRegressor

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

        window: The size of the Sakoe—Chiba band global constrant as a
            fraction of the length of the shortest of the two sequences being
            compared.

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

        Returns
        -------
        KNNRegressor
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
        """Set of possible class labels."""

        self.n_jobs: int = n_jobs
        """Maximum number of concurrently running workers."""

        self.random_state = random_state
        """Seed or :class:`numpy:numpy.random.RandomState` object for
        reproducible pseudo-randomness."""

        # Allow metadata routing for lengths
        if _sklearn.routing_enabled():
            self.set_fit_request(lengths=True)
            self.set_predict_request(lengths=True)
            self.set_score_request(lengths=True, sample_weight=True)

    def fit(
        self: KNNRegressor,
        X: FloatArray,
        y: FloatArray,
        *,
        lengths: IntArray | None = None,
    ) -> KNNRegressor:
        """Fits the regressor to the sequence(s) in ``X``.

        Parameters
        ----------
        self: KNNRegressor

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
        KNNRegressor:
            The fitted regressor.
        """
        self.X_, self.lengths_ = _validation.check_X_lengths(
            X, lengths=lengths, dtype=self._DTYPE
        )
        self.y_ = _validation.check_y(
            y,
            lengths=self.lengths_,
            dtype=np.float64,
        )
        self.idxs_ = _data.get_idxs(self.lengths_)
        self.use_c_ = _validation.check_use_c(self.use_c)
        self.random_state_ = _validation.check_random_state(self.random_state)
        _validation.check_weighting(self.weighting)
        return self

    @_validation.requires_fit
    def predict(
        self: KNNRegressor,
        X: FloatArray,
        *,
        lengths: IntArray | None = None,
    ) -> FloatArray:
        """Predicts outputs for the sequence(s) in ``X``.

        Parameters
        ----------
        self: KNNRegressor

        X:
            Sequence(s).

        lengths:
            Lengths of the sequence(s) provided in ``X``.

            - If ``None``, then ``X`` is assumed to be a single sequence.
            - ``len(X)`` should be equal to ``sum(lengths)``.

        Returns
        -------
        numpy.ndarray:
            Output predictions.

        Notes
        -----
        This method requires a trained regressor — see :func:`fit`.
        """
        _, k_distances, k_outputs = self.query_neighbors(
            X,
            lengths=lengths,
            sort=False,
        )
        k_weightings = self._weighting()(k_distances)
        total_weights = k_weightings.sum(axis=1)
        return (k_outputs * k_weightings).sum(axis=1) / total_weights
