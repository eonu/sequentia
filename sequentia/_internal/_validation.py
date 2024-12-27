# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

from __future__ import annotations

import functools
import typing as t
import warnings

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.multiclass import check_classification_targets
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import InvalidParameterError
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import NotFittedError

from sequentia._internal._typing import Array, FloatArray, IntArray

__all__ = [
    "check_random_state",
    "check_is_fitted",
    "requires_fit",
    "check_classes",
    "check_X",
    "check_X_lengths",
    "check_y",
    "check_weighting",
    "check_use_c",
]


def check_is_fitted(
    estimator: BaseEstimator,
    *,
    attributes: list[str] | None = None,
    return_: bool = False,
) -> bool | None:
    fitted = False
    if attributes is None:
        keys = estimator.__dict__
        fitted = any(attr.endswith("_") for attr in keys if "__" not in attr)
    else:
        fitted = all(hasattr(estimator, attr) for attr in attributes)

    if return_:
        return fitted

    if not fitted:
        msg = (
            f"This {type(estimator).__name__!r} instance is not fitted yet. "
            "Call 'fit' with appropriate arguments before using this method."
        )
        raise NotFittedError(msg)

    return None


def requires_fit(function: t.Callable) -> t.Callable:
    @functools.wraps(function)
    def wrapper(self: t.Self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        check_is_fitted(self)
        return function(self, *args, **kwargs)

    return wrapper


def check_classes(
    y: t.Iterable,
    *,
    classes: t.Iterable[int] | None = None,
) -> IntArray:
    check_classification_targets(y)
    unique_y = unique_labels(y)

    classes_ = None
    if classes is None:
        classes_ = unique_y
    else:
        classes_np = np.array(classes).flatten()
        if not np.issubdtype(classes_np.dtype, np.integer):
            msg = "Expected classes to be integers"
            raise TypeError(msg)

        _, idx = np.unique(classes_np, return_index=True)
        classes_ = classes_np[np.sort(idx)]
        if unseen_labels := set(unique_y) - set(classes_np):
            msg = (
                "Encountered label(s) in `y`"
                f"not present in specified classes - {unseen_labels}"
            )
            raise ValueError(msg)

    return classes_.astype(np.int8)


def check_X(
    X: t.Iterable[int] | t.Iterable[float],
    /,
    *,
    dtype: np.float64 | np.int64,
    univariate: bool = False,
) -> Array:
    if not isinstance(X, np.ndarray):
        try:
            X = np.array(X).astype(dtype)
        except Exception as e:  # noqa: BLE001
            type_ = type(X).__name__
            msg = f"Expected value to be a numpy.ndarray, got {type_!r}"
            raise TypeError(msg) from e
    if (dtype_ := X.dtype) != dtype:
        try:
            X = X.astype(dtype)
        except Exception as e:  # noqa: BLE001
            msg = f"Expected array to have dtype {dtype}, got {dtype_}"
            raise TypeError(msg) from e
    if (ndim_ := X.ndim) != 2:
        msg = f"Expected array to have two dimensions, got {ndim_}"
        raise ValueError(msg)
    if (len_ := len(X)) == 0:
        msg = f"Expected array to have be at least length 1, got {len_}"
        raise ValueError(msg)
    if univariate and (n_features := X.shape[-1]) > 1:
        msg = f"Expected array to be univariate, got {n_features} features"
        raise ValueError(msg)
    return X


def check_X_lengths(
    X: t.Iterable[int] | t.Iterable[float],
    /,
    *,
    lengths: t.Iterable[int] | None,
    dtype: np.float64 | np.int64,
    univariate: bool = False,
) -> tuple[Array, IntArray]:
    # validate observations
    X = check_X(X, dtype=dtype, univariate=univariate)

    # treat whole input as one sequence if no lengths given
    if lengths is None:
        lengths = [len(X)]

    # convert to numpy.ndarray and cast to integer
    lengths = np.array(lengths).astype(int)

    # check that there is at least one sequence
    if len(lengths) == 0:
        msg = "Expected at least one sequence"
        raise ValueError(msg)

    # check that lengths are one-dimensional
    if (ndim := lengths.ndim) != 1:
        msg = f"Expected lengths to have one dimension, got {ndim}"
        raise ValueError(msg)

    # validate sequence lengths
    if (true_total := len(X)) != (given_total := lengths.sum()):
        msg = (
            f"Total of provided lengths ({given_total}) "
            f"does not match the length of X ({true_total})"
        )
        raise ValueError(msg)

    return X, lengths


def check_y(
    y: t.Iterable[int] | t.Iterable[float] | None,
    /,
    *,
    lengths: IntArray,
    dtype: np.float64 | np.int64 | None = None,
) -> Array:
    if y is None:
        msg = "No output values `y` provided"
        raise InvalidParameterError(msg)

    # convert to numpy.ndarray and flatten
    y = np.array(y).flatten()

    # cast to dtype
    if dtype:
        y = y.astype(dtype)

    # validate against lengths
    if (len_y := len(y)) != (n_seqs := len(lengths)):
        msg = (
            f"Expected size of y ({len_y}) "
            f"to be the same as the size of lengths ({n_seqs})"
        )
        raise ValueError(msg)

    return y


def check_weighting(
    weighting: t.Callable[[FloatArray], FloatArray] | None,
    /,
) -> None:
    if weighting is None:
        return
    try:
        x = np.random.rand(10)
        weights = weighting(x)
        if not isinstance(weights, np.ndarray):
            msg = "Weights should be an numpy.ndarray"
            raise TypeError(msg)  # noqa: TRY301
        if not np.issubdtype(weights.dtype, np.floating):
            msg = "Weights should be floating point values"
            raise TypeError(msg)  # noqa: TRY301
        if x.shape != weights.shape:
            msg = "Weights should have the same shape as inputs"
            raise ValueError(msg)  # noqa: TRY301
    except Exception as e:  # noqa: BLE001
        msg = "Invalid weighting function"
        raise ValueError(msg) from e


def check_use_c(use_c: bool, /) -> bool:  # noqa: FBT001
    if not use_c:
        return use_c

    import importlib

    if importlib.util.find_spec("dtaidistance.dtw_cc"):
        return True

    msg = "DTAIDistance C library not available - using Python implementation"
    warnings.warn(msg, ImportWarning, stacklevel=1)
    return False
