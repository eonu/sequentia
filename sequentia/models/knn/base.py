# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Generic mixin class for k-nearest neigbour based models."""

from __future__ import annotations

import marshal
import pathlib
import types
import typing as t

import dtaidistance.dtw
import dtaidistance.dtw_ndim
import joblib
import numpy as np

from sequentia._internal import _data, _multiprocessing, _validation
from sequentia._internal._typing import Array, FloatArray, IntArray

__all__ = ["KNNMixin"]


class KNNMixin:
    """Generic mixin class for k-nearest neigbour based models."""

    _DTYPE: type = np.float64

    @_validation.requires_fit
    def query_neighbors(
        self: KNNMixin,
        X: FloatArray,
        *,
        lengths: IntArray | None = None,
        sort: bool = True,
    ) -> tuple[IntArray, FloatArray, Array]:
        """Query the k-nearest training observation sequences to each
        sequence in ``X``.

        Parameters
        ----------
        self: KNNMixin

        X:
            Sequence(s).

        lengths:
            Lengths of the sequence(s) provided in ``X``.

            - If ``None``, then ``X`` is assumed to be a single sequence.
            - ``len(X)`` should be equal to ``sum(lengths)``.

        sort:
            Whether to sort the neighbors in order of nearest to furthest.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
            K-nearest neighbors for each sequence in ``X``.

            - Indices of the k-nearest training sequences.
            - DTW distances of the k-nearest training sequences.
            - Corresponding outputs of the k-nearest training sequences.

        Notes
        -----
        This method requires a trained model — see :func:`fit`.
        """
        distances = self.compute_distance_matrix(X, lengths=lengths)
        if distances.shape[1] == 1:
            # only one training sequence
            # (return for all query sequences)
            k_idxs = np.zeros_like(distances, dtype=int)
            k_distances = distances
        else:
            if distances.shape[0] == 1:
                # only one query sequence
                # (use np.argsort instead of np.argpartition)
                k_idxs = distances.argsort(axis=1)[:, : self.k]
            else:
                # multiple query/training sequences
                # (use np.argpartition)
                partition_by = range(self.k) if sort else self.k
                k_idxs = np.argpartition(
                    distances,
                    partition_by,
                    axis=1,
                )[:, : self.k]
            k_distances = np.take_along_axis(distances, k_idxs, axis=1)
        k_outputs = self.y_[k_idxs]
        return k_idxs, k_distances, k_outputs

    @_validation.requires_fit
    def compute_distance_matrix(
        self: KNNMixin,
        X: FloatArray,
        *,
        lengths: IntArray | None = None,
    ) -> FloatArray:
        """Calculate a matrix of DTW distances between the sequences in
        ``X`` and the training sequences.

        Parameters
        ----------
        self: KNNMixin

        X:
            Sequence(s).

        lengths:
            Lengths of the sequence(s) provided in ``X``.

            - If ``None``, then ``X`` is assumed to be a single sequence.
            - ``len(X)`` should be equal to ``sum(lengths)``.

        Returns
        -------
        numpy.ndarray:
            DTW distance matrix.

        Notes
        -----
        This method requires a trained model — see :func:`fit`.
        """
        # validate input
        X, lengths = _validation.check_X_lengths(
            X,
            lengths=lengths,
            dtype=self._DTYPE,
        )

        # get number of jobs
        n_jobs = _multiprocessing.effective_n_jobs(self.n_jobs, x=lengths)

        # get DTW callable
        dtw = self._dtw()

        # prepare indices for multiprocessed DTW calculation
        row_chunk_idxs = np.array_split(_data.get_idxs(lengths), n_jobs)
        col_chunk_idxs = np.array_split(self.idxs_, n_jobs)

        # multiprocessed DTW calculation
        return np.vstack(
            joblib.Parallel(n_jobs=n_jobs, max_nbytes=None)(
                joblib.delayed(self._distance_matrix_row_chunk)(
                    row_idxs, col_chunk_idxs, X, n_jobs, dtw
                )
                for row_idxs in row_chunk_idxs
            )
        )

    @_validation.requires_fit
    def dtw(self: KNNMixin, A: FloatArray, B: FloatArray) -> float:
        """Calculate the DTW distance between two observation sequences.

        Parameters
        ----------
        self: KNNMixin

        A:
            The first sequence.

        B:
            The second sequence.

        Returns
        -------
        numpy.ndarray:
            DTW distance.

        Notes
        -----
        This method requires a trained model — see :func:`fit`.
        """
        A = _validation.check_X(A, dtype=self._DTYPE)
        B = _validation.check_X(B, dtype=self._DTYPE)
        return self._dtw()(A, B)

    def _dtw1d(
        self: KNNMixin,
        a: FloatArray,
        b: FloatArray,
        *,
        window: int,
    ) -> float:
        """Compute the DTW distance between two univariate sequences."""
        return dtaidistance.dtw.distance(
            a,
            b,
            use_c=self.use_c_,
            window=window,
        )

    def _window(self: KNNMixin, A: FloatArray, B: FloatArray) -> int:
        """Calculate the absolute DTW window size."""
        return int(self.window * min(len(A), len(B)))

    def _dtwi(self: KNNMixin, A: FloatArray, B: FloatArray) -> float:
        """Compute the multivariate DTW distance as the sum of the pairwise
        per-feature DTW distances, allowing each feature to be warped
        independently.
        """
        window = self._window(A, B)

        def dtw(a: FloatArray, b: FloatArray) -> float:
            """Windowed DTW wrapper function."""
            return self._dtw1d(a, b, window=window)

        return np.sum([dtw(A[:, i], B[:, i]) for i in range(A.shape[1])])

    def _dtwd(self: KNNMixin, A: FloatArray, B: FloatArray) -> float:
        """Compute the multivariate DTW distance so that the warping of the
        features depends on each other, by modifying the local distance
        measure.
        """
        window = self._window(A, B)
        return dtaidistance.dtw_ndim.distance(
            A,
            B,
            use_c=self.use_c_,
            window=window,
        )

    def _dtw(self: KNNMixin) -> t.Callable[[FloatArray], float]:
        """Conditional DTW callable."""
        return self._dtwi if self.independent else self._dtwd

    def _weighting(self: KNNMixin) -> t.Callable[[FloatArray], FloatArray]:
        """Weighting function - use equal weighting if not provided."""
        if callable(self.weighting):
            return self.weighting
        return np.ones_like

    def _distance_matrix_row_chunk(
        self: KNNMixin,
        row_idxs: IntArray,
        col_chunk_idxs: list[IntArray],
        X: FloatArray,
        n_jobs: int,
        dtw: t.Callable[[FloatArray], float],
    ) -> FloatArray:
        """Calculate a distance sub-matrix for a subset of rows over all
        columns.
        """
        return np.hstack(
            joblib.Parallel(n_jobs=n_jobs, max_nbytes=None)(
                joblib.delayed(self._distance_matrix_row_col_chunk)(
                    col_idxs, row_idxs, X, dtw
                )
                for col_idxs in col_chunk_idxs
            )
        )

    def _distance_matrix_row_col_chunk(
        self: KNNMixin,
        col_idxs: IntArray,
        row_idxs: IntArray,
        X: FloatArray,
        dtw: t.Callable[[FloatArray], float],
    ) -> FloatArray:
        """Calculate a distance sub-matrix for a subset of rows and
        columns.
        """
        distances = np.zeros((len(row_idxs), len(col_idxs)))
        for i, x_row in enumerate(_data.iter_X(X, idxs=row_idxs)):
            for j, x_col in enumerate(_data.iter_X(self.X_, idxs=col_idxs)):
                distances[i, j] = dtw(x_row, x_col)
        return distances

    @_validation.requires_fit
    def save(
        self: KNNMixin,
        path: str | pathlib.Path | t.IO,
        /,
    ) -> None:
        """Serialize and save a fitted KNN estimator.

        Parameters
        ----------
        self: KNNMixin

        path:
            Location to save the serialized estimator.

        Notes
        -----
        This method requires a trained model — see :func:`fit`.

        See Also
        --------
        load:
            Load and deserialize a fitted KNN estimator.
        """
        # fetch main parameters and fitted values
        dict_ = self.__dict__.items()
        state = {
            "params": self.get_params(),
            "fitted": {k: v for k, v in dict_ if k.endswith("_")},
        }

        # serialize weighting function
        if self.weighting is None:
            state["params"]["weighting"] = self.weighting
        else:
            state["params"]["weighting"] = marshal.dumps(
                (self.weighting.__code__, self.weighting.__name__)
            )

        # serialize model
        joblib.dump(state, path)

    @classmethod
    def load(
        cls: type[KNNMixin],
        path: str | pathlib.Path | t.IO,
        /,
    ) -> KNNMixin:
        """Load and deserialize a fitted KNN estimator.

        Parameters
        ----------
        cls: type[KNNMixin]

        path:
            Location to load the serialized estimator from.

        Returns
        -------
        KNNMixin:
            Fitted KNN estimator.

        See Also
        --------
        save:
            Serialize and save a fitted KNN estimator.
        """
        state = joblib.load(path)

        # deserialize weighting function
        if state["params"]["weighting"] is not None:
            weighting_ = state["params"]["weighting"]
            weighting, name = marshal.loads(weighting_)  # noqa: S302
            state["params"]["weighting"] = types.FunctionType(
                weighting, globals(), name
            )

        # set main parameters
        model = cls(**state["params"])

        # set fitted values
        for k, v in state["fitted"].items():
            setattr(model, k, v)

        # return deserialized model
        return model
