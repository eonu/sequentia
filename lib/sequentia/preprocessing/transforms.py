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
from typing import Callable, Optional, Dict, Any

import numpy as np
from pydantic import PositiveInt
from sklearn.utils.validation import _allclose_dense_sparse, check_array
from scipy.signal import medfilt2d, convolve

from sequentia.preprocessing.base import Transform
from sequentia.utils.validation import _BaseSequenceValidator, Array
from sequentia.utils.data import SequentialDataset

__all__ = ["IndependentFunctionTransformer", "mean_filter", "median_filter"]


class IndependentFunctionTransformer(Transform):
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
        Xt = transform.transform(data.X, data.lengths)
    """


    def __init__(
        self,
        func: Optional[Callable] = None,
        inverse_func: Optional[Callable] = None,
        *,
        validate: bool = False,
        check_inverse: bool = True,
        kw_args =None,
        inv_kw_args=None,
    ) -> IndependentFunctionTransformer:
        """Initializes the :class:`.IndependentFunctionTransformer`.

        :param func: The callable to use for the transformation.
            This will be passed the same arguments as transform, with args and kwargs forwarded.
            If ``None``, then ``func`` will be the identity function.

        :param inverse_func: The callable to use for the inverse transformation.
            This will be passed the same arguments as inverse transform, with args and kwargs forwarded.
            If ``None``, then ``inverse_func`` will be the identity function.

        :param validate: Indicates whether the input ``X`` array should be checked before calling ``func``.

            - If ``False``, there is no input validation.
            - If ``True``, then ``X`` will be converted to a 2-dimensional NumPy array.
              If the conversion is not possible an exception is raised.

        :param check_inverse: Whether to check that or ``func`` followed by ``inverse_func`` leads to the original inputs.
            It can be used for a sanity check, raising a warning when the condition is not fulfilled.

        :param kw_args: Dictionary of additional keyword arguments to pass to ``func``.

        :param inv_kw_args: Dictionary of additional keyword arguments to pass to ``inverse_func``.
        """
        self.func = func
        self.inverse_func = inverse_func
        self.validate = validate
        self.check_inverse = check_inverse
        self.kw_args: Optional[Dict[str, Any]] = kw_args
        self.inv_kw_args: Optional[Dict[str, Any]] = inv_kw_args

    def _check_input(self, X, lengths):
        data = _BaseSequenceValidator(X=X, lengths=lengths)
        return data.X, data.lengths

    def _check_inverse_transform(self, X, lengths):
        """Check that func and inverse_func are the inverse."""
        idx_selected = slice(None, None, max(1, X.shape[0] // 100))
        X_round_trip = self.inverse_transform(self.transform(X[idx_selected], lengths), lengths)

        if not np.issubdtype(X.dtype, np.number):
            raise ValueError(
                "'check_inverse' is only supported when all the elements in `X` are numerical."
            )

        if not _allclose_dense_sparse(X[idx_selected], X_round_trip):
            warnings.warn(
                "The provided functions are not strictly"
                " inverse of each other. If you are sure you"
                " want to proceed regardless, set"
                " 'check_inverse=False'.",
                UserWarning,
            )

    def fit(
        self,
        X: Array,
        lengths: Optional[Array] = None
    ) -> IndependentFunctionTransformer:
        """Fits the transformer to ``X``.

        :param X: Univariate or multivariate observation sequence(s).

            - Should be a single 1D or 2D array.
            - Should have length as the 1st dimension and features as the 2nd dimension.
            - Should be a concatenated sequence if multiple sequences are provided,
              with respective sequence lengths being provided in the ``lengths`` argument for decoding the original sequences.

        :param lengths: Lengths of the observation sequence(s) provided in ``X``.

            - If ``None``, then ``X`` is assumed to be a single observation sequence.
            - ``len(X)`` should be equal to ``sum(lengths)``.

        :return: The fitted transformer.
        """
        X, lengths = self._check_input(X, lengths)
        if self.check_inverse and not (self.func is None or self.inverse_func is None):
            self._check_inverse_transform(X, lengths)
        return self

    def transform(
        self,
        X: Array,
        lengths: Optional[Array] = None
    ) -> Array:
        """Applies the transformation to ``X``, producing a transformed version of ``X``.

        :param X: Univariate or multivariate observation sequence(s).

            - Should be a single 1D or 2D array.
            - Should have length as the 1st dimension and features as the 2nd dimension.
            - Should be a concatenated sequence if multiple sequences are provided,
              with respective sequence lengths being provided in the ``lengths`` argument for decoding the original sequences.

        :param lengths: Lengths of the observation sequence(s) provided in ``X``.

            - If ``None``, then ``X`` is assumed to be a single observation sequence.
            - ``len(X)`` should be equal to ``sum(lengths)``.

        :return: The transformed array.
        """
        X, lengths = self._check_input(X, lengths)
        return self._transform(X, lengths, func=self.func, kw_args=self.kw_args)

    def inverse_transform(
        self,
        X: Array,
        lengths: Optional[Array] = None
    ) -> Array:
        """Applies the inverse transformation to ``X``.

        :param X: Univariate or multivariate observation sequence(s).

            - Should be a single 1D or 2D array.
            - Should have length as the 1st dimension and features as the 2nd dimension.
            - Should be a concatenated sequence if multiple sequences are provided,
              with respective sequence lengths being provided in the ``lengths`` argument for decoding the original sequences.

        :param lengths: Lengths of the observation sequence(s) provided in ``X``.

            - If ``None``, then ``X`` is assumed to be a single observation sequence.
            - ``len(X)`` should be equal to ``sum(lengths)``.

        :return: The inverse transformed array.
        """
        X, lengths = self._check_input(X, lengths)
        if self.validate:
            X = check_array(X, accept_sparse=False)
        return self._transform(X, lengths, func=self.inverse_func, kw_args=self.inv_kw_args)

    def _transform(self, X, lengths, func=None, kw_args=None):
        if func is None:
            return X
        apply = lambda x: func(x, **(kw_args if kw_args else {}))
        idxs = SequentialDataset._get_idxs(lengths)
        return np.vstack([apply(x) for x in SequentialDataset._iter_X(X, idxs)])

    def __sklearn_is_fitted__(self):
        """Return True since FunctionTransfomer is stateless."""
        return True

    def _more_tags(self):
        return {"no_validation": not self.validate, "stateless": True}


def mean_filter(x: Array, k: PositiveInt = 5) -> Array:
    """Applies a mean filter of size ``k`` independently to each feature of the sequence,
    retaining the original input shape by using appropriate padding.

    This is implemented as a 1D convolution with a kernel of size ``k`` and values ``1 / k``.

    :param x: Univariate or multivariate observation sequence.

        - Should be a single 1D or 2D array.
        - Should have length as the 1st dimension and features as the 2nd dimension.

    :param k: Width of the filter.

    :return: The filtered array.

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
        Xt = transform.transform(data.X, data.lengths)
    """
    data = _BaseSequenceValidator(X=x)
    return convolve(data.X, np.ones((k, 1)) / k, mode="same")


def median_filter(x: Array, k: PositiveInt = 5) -> Array:
    """Applies a median filter of size ``k`` independently to each feature of the sequence,
    retaining the original input shape by using appropriate padding.

    :param x: Univariate or multivariate observation sequence.

        - Should be a single 1D or 2D array.
        - Should have length as the 1st dimension and features as the 2nd dimension.

    :param k: Width of the filter.

    :return: The filtered array.

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
        Xt = transform.transform(data.X, data.lengths)
    """
    data = _BaseSequenceValidator(X=x)
    return medfilt2d(data.X, kernel_size=(k, 1))
