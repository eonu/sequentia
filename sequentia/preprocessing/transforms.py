# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""
IndependentFunctionTransformer is an adapted version of FunctionTransformer
from the sklearn.preprocessing module, and largely relies on its source code.

Below is the original license from Scikit-Learn, copied on 31st December 2022
from https://github.com/scikit-learn/scikit-learn/blob/main/COPYING.

---

BSD 3-Clause License

Copyright (c) 2007-2022 The scikit-learn developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from __future__ import annotations

import warnings

import numpy as np
import scipy.signal
import sklearn
import sklearn.base
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.validation import _allclose_dense_sparse, check_array

from sequentia._internal import _data, _sklearn, _validation
from sequentia._internal._typing import Array, FloatArray, IntArray

__all__ = ["IndependentFunctionTransformer", "mean_filter", "median_filter"]


class IndependentFunctionTransformer(FunctionTransformer):
    """Constructs a transformer from an arbitrary callable,
    applying the transform independently to each sequence.

    This transform forwards its ``X`` and ``lengths`` arguments
    to a user-defined function or function object and returns the result of this
    function. This is useful for stateless transformations such as taking the
    log of frequencies, doing custom scaling, etc.
    Note: If a lambda is used as the function, then the resulting
    transformer will not be pickleable.

    This works conveniently with functions in :mod:`sklearn.preprocessing`
    such as :func:`~sklearn.preprocessing.scale` or :func:`~sklearn.preprocessing.normalize`.

    :note: This is a stateless transform, meaning :func:`fit` and :func:`fit_transform` will not fit on any data.

    See Also
    --------
    :class:`sklearn.preprocessing.FunctionTransformer`
        :class:`.IndependentFunctionTransformer` is based on this class,
        which applies the callable to the entire input array ``X`` as if it were a single sequence.
        Read more in the :ref:`User Guide <function_transformer>`.

    Examples
    --------
    Using an :class:`IndependentFunctionTransformer` with :func:`sklearn.preprocessing.minmax_scale` to
    scale features to the range [0, 1] independently for each sequence in the spoken digits dataset. ::

        from sklearn.preprocessing import minmax_scale
        from sequentia.preprocessing import IndependentFunctionTransformer
        from sequentia.datasets import load_digits

        # Fetch MFCCs of spoken digits
        data = load_digits()

        # Create an independent min-max transform
        transform = IndependentFunctionTransformer(minmax_scale)

        # Apply the transform to the data
        Xt = transform.transform(data.X, lengths=data.lengths)
    """

    def __init__(
        self,
        func=None,
        inverse_func=None,
        *,
        validate=False,
        accept_sparse=False,
        check_inverse=True,
        feature_names_out=None,
        kw_args=None,
        inv_kw_args=None,
    ):
        """See :class:`sklearn:sklearn.preprocessing.FunctionTransformer`."""
        self.func = func
        self.inverse_func = inverse_func
        self.validate = validate
        self.accept_sparse = accept_sparse
        self.check_inverse = check_inverse
        self.feature_names_out = feature_names_out
        self.kw_args = kw_args
        self.inv_kw_args = inv_kw_args

        # Allow metadata routing for lengths
        if _sklearn.routing_enabled():
            self.set_fit_request(lengths=True)
            self.set_transform_request(lengths=True)
            self.set_inverse_transform_request(lengths=True)

    def _check_input(self, X, *, lengths, reset):
        if self.validate:
            X, lengths = _validation.check_X_lengths(
                X, lengths=lengths, dtype=X.dtype
            )
            return (
                self._validate_data(
                    X, accept_sparse=self.accept_sparse, reset=reset
                ),
                lengths,
            )
        return X, lengths

    def _check_inverse_transform(self, X, *, lengths):
        """Check that func and inverse_func are the inverse."""
        idx_selected = slice(None, None, max(1, X.shape[0] // 100))
        X_round_trip = self.inverse_transform(
            self.transform(X[idx_selected], lengths=lengths),
            lengths=lengths,
        )

        if hasattr(X, "dtype"):
            dtypes = [X.dtype]
        elif hasattr(X, "dtypes"):
            # Dataframes can have multiple dtypes
            dtypes = X.dtypes

        if not all(np.issubdtype(d, np.number) for d in dtypes):
            raise ValueError(
                "'check_inverse' is only supported when all the elements in `X` is"
                " numerical."
            )

        if not _allclose_dense_sparse(X[idx_selected], X_round_trip):
            warnings.warn(
                (
                    "The provided functions are not strictly"
                    " inverse of each other. If you are sure you"
                    " want to proceed regardless, set"
                    " 'check_inverse=False'."
                ),
                UserWarning,
            )

    @sklearn.base._fit_context(prefer_skip_nested_validation=True)
    def fit(
        self: IndependentFunctionTransformer,
        X: Array,
        y: Array | None = None,
        *,
        lengths: IntArray | None = None,
    ) -> IndependentFunctionTransformer:
        """Fits the transformer to ``X``.

        Parameters
        ----------
        self: IndependentFunctionTransformer

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
        IndependentFunctionTransformer
            The fitted transformer.
        """
        X, lengths = self._check_input(X, lengths=lengths, reset=True)
        if self.check_inverse and not (
            self.func is None or self.inverse_func is None
        ):
            self._check_inverse_transform(X, lengths=lengths)
        return self

    def transform(
        self: IndependentFunctionTransformer,
        X: Array,
        *,
        lengths: IntArray | None = None,
    ) -> Array:
        """Applies the transformation to ``X``,
        producing a transformed version of ``X``.

        Parameters
        ----------
        self: IndependentFunctionTransformer

        X:
            Sequence(s).

        lengths:
            Lengths of the sequence(s) provided in ``X``.

            - If ``None``, then ``X`` is assumed to be a single sequence.
            - ``len(X)`` should be equal to ``sum(lengths)``.

        Returns
        -------
        numpy.ndarray:
            The transformed array.
        """
        X, lengths = self._check_input(X, lengths=lengths, reset=False)
        return self._transform(
            X, lengths=lengths, func=self.func, kw_args=self.kw_args
        )

    def inverse_transform(
        self: IndependentFunctionTransformer,
        X: Array,
        *,
        lengths: IntArray | None = None,
    ) -> Array:
        """Applies the inverse transformation to ``X``.

        Parameters
        ----------
        self: IndependentFunctionTransformer

        X:
            Sequence(s).

        lengths:
            Lengths of the sequence(s) provided in ``X``.

            - If ``None``, then ``X`` is assumed to be a single sequence.
            - ``len(X)`` should be equal to ``sum(lengths)``.

        Returns
        -------
        numpy.ndarray:
            The inverse transformed array.
        """
        if self.validate:
            X = check_array(X, accept_sparse=False)
            X, lengths = _validation.check_X_lengths(
                X, lengths=lengths, dtype=X.dtype
            )
        return self._transform(
            X,
            lengths=lengths,
            func=self.inverse_func,
            kw_args=self.inv_kw_args,
        )

    def fit_transform(
        self: IndependentFunctionTransformer,
        X: Array,
        y: Array | None = None,
        *,
        lengths: IntArray | None = None,
    ) -> Array:
        """Fits the transformer to the sequence(s) in ``X`` and returns a
        transformed version of ``X``.

        Parameters
        ----------
        self: IndependentFunctionTransformer

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
            The transformed data.
        """
        return self.fit(X, lengths=lengths).transform(X, lengths=lengths)

    def _transform(self, X, *, lengths, func=None, kw_args=None):
        if func is None:
            return X
        apply = lambda x: func(x, **(kw_args if kw_args else {}))
        idxs = _data.get_idxs(lengths)
        return np.vstack([apply(x) for x in _data.iter_X(X, idxs=idxs)])


def mean_filter(x: FloatArray, *, k: int = 5) -> FloatArray:
    """Applies a mean filter of size ``k`` independently to each feature of
    the sequence, retaining the original input shape by using appropriate
    padding.

    This is implemented as a 1D convolution with a kernel of size ``k`` and
    values ``1 / k``.

    Parameters
    ----------
    x:
        Observation sequence.

    k:
        Width of the filter.

    Returns
    -------
    numpy.ndarray:
        The filtered array.

    Examples
    --------
    Applying a :func:`mean_filter` to a single sequence
    and multiple sequences (independently via :class:`IndependentFunctionTransformer`) from the spoken digits dataset. ::

        from sequentia.preprocessing import IndependentFunctionTransformer, mean_filter
        from sequentia.datasets import load_digits

        # Fetch MFCCs of spoken digits
        data = load_digits()

        # Apply the mean filter to the first sequence
        x, _ = data[0]
        xt = mean_filter(x, k=7)

        # Create an independent mean filter transform
        transform = IndependentFunctionTransformer(mean_filter, kw_args={"k": 7})

        # Apply the transform to all sequences
        Xt = transform.transform(data.X, lengths=data.lengths)
    """
    return scipy.signal.convolve(x, np.ones((k, 1)) / k, mode="same")


def median_filter(x: FloatArray, *, k: int = 5) -> FloatArray:
    """Applies a median filter of size ``k`` independently to each feature of
    the sequence, retaining the original input shape by using appropriate
    padding.

    Parameters
    ----------
    x:
        Observation sequence.

    k:
        Width of the filter.

    Returns
    -------
    numpy.ndarray:
        The filtered array.

    Examples
    --------
    Applying a :func:`median_filter` to a single sequence
    and multiple sequences (independently via :class:`IndependentFunctionTransformer`) from the spoken digits dataset. ::

        from sequentia.preprocessing import IndependentFunctionTransformer, median_filter
        from sequentia.datasets import load_digits

        # Fetch MFCCs of spoken digits
        data = load_digits()

        # Apply the median filter to the first sequence
        x, _ = data[0]
        xt = median_filter(x, k=7)

        # Create an independent median filter transform
        transform = IndependentFunctionTransformer(median_filter, kw_args={"k": 7})

        # Apply the transform to all sequences
        Xt = transform.transform(data.X, lengths=data.lengths)
    """
    return scipy.signal.medfilt2d(x, kernel_size=(k, 1))
