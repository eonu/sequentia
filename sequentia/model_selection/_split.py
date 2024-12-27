# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

import typing as t

import numpy as np
from sklearn.model_selection import _split

__all__ = [
    "KFold",
    "StratifiedKFold",
    "ShuffleSplit",
    "StratifiedShuffleSplit",
    "RepeatedKFold",
    "RepeatedStratifiedKFold",
]


class KFold(_split.KFold):
    """K-Fold cross-validator.

    Provides train/test indices to split data in train/test sets.
    Split dataset into k consecutive folds (without shuffling by default).

    Each fold is then used once as a validation while the
    k - 1 remaining folds form the training set.

    See Also
    --------
    :class:`sklearn.model_selection.KFold`
        :class:`.KFold` is a modified version
        of this class that supports sequences.
    """

    def split(
        self, X: np.ndarray, y: np.ndarray, groups: t.Any = None
    ) -> None:
        return super().split(y, y, groups)


class StratifiedKFold(_split.StratifiedKFold):
    """Stratified K-Fold cross-validator.

    Provides train/test indices to split data in train/test sets.

    This cross-validation object is a variation of
    KFold that returns stratified folds.

    The folds are made by preserving the percentage of samples for each class.

    See Also
    --------
    :class:`sklearn.model_selection.StratifiedKFold`
        :class:`.StratifiedKFold` is a modified version
        of this class that supports sequences.
    """

    def split(
        self, X: np.ndarray, y: np.ndarray, groups: t.Any = None
    ) -> None:
        return super().split(y, y, groups)


class ShuffleSplit(_split.ShuffleSplit):
    """Random permutation cross-validator.

    Yields indices to split data into training and test sets.

    Note: contrary to other cross-validation strategies, random splits do not
    guarantee that test sets across all folds will be mutually exclusive,
    and might include overlapping samples. However, this is still very likely
    for sizeable datasets.

    See Also
    --------
    :class:`sklearn.model_selection.ShuffleSplit`
        :class:`.ShuffleSplit` is a modified version
        of this class that supports sequences.
    """

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        groups: t.Any = None,
    ) -> None:
        return super().split(y, y, groups)


class StratifiedShuffleSplit(_split.StratifiedShuffleSplit):
    """Stratified :class:`.ShuffleSplit` cross-validator.

    Provides train/test indices to split data in train/test sets.

    This cross-validation object is a merge of :class:`.StratifiedKFold`
    and :class:`.ShuffleSplit`, which returns stratified randomized folds.
    The folds are made by preserving the percentage of samples for each class.

    See Also
    --------
    :class:`sklearn.model_selection.StratifiedShuffleSplit`
        :class:`.StratifiedShuffleSplit` is a modified version
        of this class that supports sequences.
    """

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        groups: t.Any = None,
    ) -> None:
        return super().split(y, y, groups)


class RepeatedKFold(_split.RepeatedKFold):
    """Repeated :class:`.KFold` cross validator.

    Repeats :class:`.KFold` n times with different randomization in each repetition.

    See Also
    --------
    :class:`sklearn.model_selection.RepeatedKFold`
        :class:`.RepeatedKFold` is a modified version
        of this class that supports sequences.
    """

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        groups: t.Any = None,
    ) -> None:
        return super().split(y, y, groups)


class RepeatedStratifiedKFold(_split.RepeatedStratifiedKFold):
    """Repeated :class:`.StratifiedKFold` cross validator.

    Repeats :class:`.StratifiedKFold` n times with different randomization
    in each repetition.

    See Also
    --------
    :class:`sklearn.model_selection.RepeatedStratifiedKFold`
        :class:`.RepeatedStratifiedKFold` is a modified version
        of this class that supports sequences.
    """

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        groups: t.Any = None,
    ) -> None:
        return super().split(y, y, groups)
