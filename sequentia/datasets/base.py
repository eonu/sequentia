# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Utility wrapper for a generic sequential dataset."""

from __future__ import annotations

import copy
import pathlib
import typing as t
import warnings

import numpy as np
import pydantic as pyd
from sklearn.model_selection import train_test_split

from sequentia._internal import _data, _validation
from sequentia._internal._typing import Array, IntArray

__all__ = ["SequentialDataset"]


class SequentialDataset:
    """Utility wrapper for a generic sequential dataset."""

    def __init__(
        self: SequentialDataset,
        X: Array,
        y: Array | None = None,
        *,
        lengths: IntArray | None = None,
        classes: list[int] | None = None,
    ) -> SequentialDataset:
        """Initialize a :class:`.SequentialDataset`.

        Parameters
        ----------
        self: SequentialDataset

        X:
            Sequence(s).

        y:
            Outputs corresponding to sequence(s) in ``X``.

        lengths:
            Lengths of the sequence(s) provided in ``X``.

            - If ``None``, then ``X`` is assumed to be a single sequence.
            - ``len(X)`` should be equal to ``sum(lengths)``.

        classes:
            Set of possible class labels
            (only if ``y`` was provided with categorical values).

            If not provided, these will be determined from the training
            data labels.
        """
        X, lengths = _validation.check_X_lengths(
            X,
            lengths=lengths,
            dtype=X.dtype,
        )
        if y is not None:
            y = _validation.check_y(y, lengths=lengths)

        self._X = X
        self._y = y
        self._lengths = lengths

        self._classes = None
        if self._y is not None and np.issubdtype(self._y.dtype, np.integer):
            self._classes = _validation.check_classes(
                self._y,
                classes=classes,
            )

        self._idxs = _data.get_idxs(self.lengths)

    def split(
        self: SequentialDataset,
        *,
        test_size: (
            pyd.NonNegativeInt | pyd.confloat(ge=0, le=1) | None
        ) = None,  # placeholder
        train_size: (
            pyd.NonNegativeInt | pyd.confloat(ge=0, le=1) | None
        ) = None,  # placeholder
        random_state: (
            pyd.NonNegativeInt | np.random.RandomState | None
        ) = None,  # placeholder
        shuffle: bool = True,
        stratify: bool = False,
    ) -> tuple[SequentialDataset, SequentialDataset]:
        """Split the dataset into two partitions (train/test).

        See :func:`sklearn:sklearn.model_selection.train_test_split`.

        Parameters
        ----------
        self: SequentialDataset

        test_size:
            Size of the test partition.

        train_size:
            Size of the training partition.

        random_state:
            Seed or :class:`numpy:numpy.random.RandomState` object for
            reproducible pseudo-randomness.

        shuffle:
            Whether or not to shuffle the data before splitting.
            If ``shuffle=False`` then ``stratify`` must be ``False``.

        stratify:
            Whether or not to stratify the partitions by class label.

        Returns
        -------
        tuple[SequentialDataset, SequentialDataset]
            Dataset partitions.
        """
        stratify = None
        if stratify:
            if self._y is None:
                msg = "Cannot stratify with no provided outputs"
                warnings.warn(msg, stacklevel=1)
            elif self._classes is None:
                msg = "Cannot stratify on non-categorical outputs"
                warnings.warn(msg, stacklevel=1)
            else:
                stratify = self._y

        idxs = np.arange(len(self._lengths))
        train_idxs, test_idxs = train_test_split(
            idxs,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify,
        )

        if self._y is None:
            X_train, y_train = self[train_idxs], None
            X_test, y_test = self[test_idxs], None
        else:
            X_train, y_train = self[train_idxs]
            X_test, y_test = self[test_idxs]

        lengths_train = self._lengths[train_idxs]
        lengths_test = self._lengths[test_idxs]
        classes = self._classes

        data_train = SequentialDataset(
            np.vstack(X_train),
            y_train,
            lengths=lengths_train,
            classes=classes,
        )
        data_test = SequentialDataset(
            np.vstack(X_test),
            y_test,
            lengths=lengths_test,
            classes=classes,
        )

        return data_train, data_test

    def iter_by_class(
        self: SequentialDataset,
    ) -> t.Generator[tuple[Array, Array, int]]:
        """Subset the observation sequences by class.

        Returns
        -------
        typing.Generator[tuple[numpy.ndarray, numpy.ndarray, int]]
            Generator iterating over classes, yielding:

            - ``X`` subset of sequences belonging to the class.
            - Lengths corresponding to the ``X`` subset.
            - Class used to subset ``X``.

        Raises
        ------
        AttributeError
            If ``y`` was not provided to :func:`__init__`.

        TypeError
            If ``y`` was provided but was not categorical.
        """
        if self._y is None:
            msg = "No `y` values were provided during initialization"
            raise AttributeError(msg)

        if self._classes is None:
            msg = "Cannot iterate by class on real-valued targets"
            raise TypeError(msg)

        for c in self._classes:
            ind = np.argwhere(self._y == c).flatten()
            X, _ = self[ind]
            lengths = self._lengths[ind]
            yield np.vstack(X), lengths, c

    def __len__(self: SequentialDataset) -> int:
        """Return the number of sequences in the dataset."""
        return len(self._lengths)

    def __getitem__(
        self: SequentialDataset,
        /,
        i: int,
    ) -> Array | tuple[Array, Array]:
        """Slice observation sequences and corresponding outputs."""
        idxs = np.atleast_2d(self._idxs[i])
        X = list(_data.iter_X(self._X, idxs=idxs))
        X = X[0] if isinstance(i, int) and len(X) == 1 else X
        return X if self._y is None else (X, self._y[i])

    def __iter__(
        self: SequentialDataset,
    ) -> t.Generator[Array | tuple[Array, Array]]:
        """Create a generator over sequences and their corresponding
        outputs.
        """
        for i in range(len(self)):
            yield self[i]

    @property
    def X(self: SequentialDataset) -> Array:
        """Observation sequences.

        Returns
        -------
        numpy.ndarray
            Observation sequences.
        """
        return self._X

    @property
    def y(self: SequentialDataset) -> Array:
        """Outputs corresponding to ``X``.

        Returns
        -------
        numpy.ndarray
            Sequence outputs.

        Raises
        ------
        AttributeError
            If ``y`` was not provided to :func:`__init__`.
        """
        if self._y is None:
            msg = "No `y` values were provided during initialization"
            raise AttributeError(msg)
        return self._y

    @property
    def lengths(self: SequentialDataset) -> IntArray:
        """Lengths corresponding to ``X``.

        Returns
        -------
        numpy.ndarray
            Lengths for each sequence in ``X``.
        """
        return self._lengths

    @property
    def classes(self: SequentialDataset) -> IntArray | None:
        """Set of unique classes in ``y``.

        Returns
        -------
        numpy.ndarray | None
            Unique classes if ``y`` is categorical.
        """
        return self._classes

    @property
    def idxs(self: SequentialDataset) -> IntArray:
        """Observation sequence start and end indices.

        Returns
        -------
        numpy.ndarray
            Start and end indices for each sequence in ``X``.
        """
        return self._idxs

    @property
    def X_y(self: SequentialDataset) -> dict[str, Array]:
        """Observation sequences and corresponding outputs.

        Returns
        -------
        dict[str, numpy.ndarray]
            Mapping with keys:

            - ``"X"`` for observation sequences,
            - ``"y"`` for outputs.

        Raises
        ------
        AttributeError
            If ``y`` was not provided to :func:`__init__`.
        """
        if self._y is None:
            msg = "No `y` values were provided during initialization"
            raise AttributeError(msg)
        return {"X": self._X, "y": self._y}

    @property
    def X_lengths(self: SequentialDataset) -> dict[str, Array]:
        """Observation sequences and corresponding lengths.

        Returns
        -------
        dict[str, numpy.ndarray]
            Mapping with keys:

            - ``"X"`` for observation sequences,
            - ``"lengths"`` for lengths.
        """
        return {"X": self._X, "lengths": self._lengths}

    @property
    def X_y_lengths(self: SequentialDataset) -> dict[str, Array]:
        """Observation sequences and corresponding outputs and lengths.

        Returns
        -------
        dict[str, numpy.ndarray]
            Mapping with keys:

            - ``"X"`` for observation sequences,
            - ``"y"`` for outputs,
            - ``"lengths"`` for lengths.

        Raises
        ------
        AttributeError
            If ``y`` was not provided to :func:`__init__`.
        """
        if self._y is None:
            msg = "No `y` values were provided during initialization"
            raise AttributeError(msg)
        return {"X": self._X, "y": self._y, "lengths": self._lengths}

    def save(
        self: SequentialDataset,
        path: str | pathlib.Path | t.IO,
        /,
        *,
        compress: bool = True,
    ) -> None:
        """Store the dataset in ``.npz`` format.

        See :func:`numpy:numpy.savez` and :func:`numpy:numpy.savez_compressed`.

        Parameters
        ----------
        path
            Location to store the dataset.

        compress
            Whether or not to compress the dataset.

        See Also
        --------
        load:
            Loads a stored dataset in ``.npz`` format.
        """
        arrs = self.X_lengths

        if self._y is not None:
            arrs["y"] = self._y

        if self._classes is not None:
            arrs["classes"] = self._classes

        save_fun = np.savez_compressed if compress else np.savez
        save_fun(path, **arrs)

    @classmethod
    def load(
        cls: type[SequentialDataset], path: str | pathlib.Path | t.IO, /
    ) -> SequentialDataset:
        """Load a stored dataset in ``.npz`` format.

        See :func:`numpy:numpy.load`.

        Parameters
        ----------
        path:
            Location to store the dataset.

        Returns
        -------
        SequentialDataset
            The loaded dataset.

        See Also
        --------
        save:
            Stores the dataset in ``.npz`` format.
        """
        return cls(**np.load(path))

    def copy(self: SequentialDataset) -> SequentialDataset:
        """Create a copy of the dataset.

        Returns
        -------
        SequentialDataset
            Dataset copy.
        """
        params = {
            "X": copy.deepcopy(self._X),
            "y": None,
            "lengths": copy.deepcopy(self._lengths),
            "classes": None,
        }

        if self._y is not None:
            params["y"] = copy.deepcopy(self._y)

        if self._classes is not None:
            params["classes"] = copy.deepcopy(self._classes)

        return SequentialDataset(
            params["X"],
            params["y"],
            lengths=params["lengths"],
            classes=params["classes"],
        )
