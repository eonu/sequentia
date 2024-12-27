# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""This file is an adapted version of the same file from the
sklearn.model_selection sub-package.

Below is the original license from Scikit-Learn, copied on 27th December 2024
from https://github.com/scikit-learn/scikit-learn/blob/main/COPYING.

---

BSD 3-Clause License

Copyright (c) 2007-2024 The scikit-learn developers.
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
