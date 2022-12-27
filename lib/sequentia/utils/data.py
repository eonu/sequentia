from __future__ import annotations

import copy
import warnings
import pathlib
from typing import Optional, Tuple, Union, Iterator, IO

import numpy as np
from pydantic import NonNegativeInt, confloat
from sklearn.model_selection import train_test_split

from sequentia.utils.validation import _check_classes, _BaseSequenceValidator, Array

__all__ = ['SequentialDataset']


class SequentialDataset:
    """Utility wrapper for a generic sequential dataset."""

    def __init__(
        self,
        X: Array,
        y: Optional[Array] = None,
        lengths: Optional[Array[int]] = None,
        classes: Optional[Array[int]] = None
    ) -> SequentialDataset:
        """Initializes a :class:`.SequentialDataset`.

        :param X: Univariate or multivariate observation sequence(s).

            - Should be a single 1D or 2D array.
            - Should have length as the 1st dimension and features as the 2nd dimension.
            - Should be a concatenated sequence if multiple sequences are provided,
              with respective sequence lengths being provided in the ``lengths`` argument for decoding the original sequences.

        :param y: Outputs corresponding to sequence(s) provided in ``X``.

        :param lengths: Lengths of the observation sequence(s) provided in ``X``.

            - If ``None``, then ``X`` is assumed to be a single observation sequence.
            - ``len(X)`` should be equal to ``sum(lengths)``.

        :param classes: Set of possible class labels (only if ``y`` was provided with categorical values).

            - If not provided, these will be determined from the training data labels.
        """
        data = _BaseSequenceValidator(X=X, lengths=lengths, y=y)

        self._X = data.X
        self._y = data.y
        self._lengths = data.lengths

        self._classes = None
        if self._y is not None and np.issubdtype(self._y.dtype, np.integer):
            self._classes = _check_classes(self._y, classes)

        self._idxs = self._get_idxs(self.lengths)

        self._X_y = (self._X, self._y)
        self._X_lengths = (self._X, self._lengths)
        self._X_y_lengths = (self._X, self._y, self._lengths)


    def split(
        self,
        test_size: Optional[Union[NonNegativeInt, confloat(ge=0, le=1)]] = None,
        train_size: Optional[Union[NonNegativeInt, confloat(ge=0, le=1)]] = None,
        random_state: Optional[Union[NonNegativeInt, np.random.RandomState]] = None,
        shuffle: bool = True,
        stratify: bool = False
    ) -> Tuple[SequentialDataset, SequentialDataset]:
        """Splits the dataset into two partitions (train/test).

        See :func:`sklearn:sklearn.model_selection.train_test_split`.

        :param test_size: Size of the test partition.
        :param train_size: Size of the train partition.
        :param random_state: Seed or :class:`numpy:numpy.random.RandomState` object for reproducible pseudo-randomness.
        :param shuffle: Whether or not to shuffle the data before splitting. If ``shuffle=False`` then ``stratify`` must be ``False``.
        :param stratify: Whether or not to stratify the partitions by class labels.
        :return: Dataset partitions.
        """
        if stratify and self._y is None:
            warnings.warn('Cannot stratify with no provided outputs')
            stratify = None
        else:
            if stratify:
                if self._classes is None:
                    warnings.warn('Cannot stratify on non-categorical outputs')
                    stratify = None
                else:
                    stratify = self._y
            else:
                stratify = None

        idxs = np.arange(len(self._lengths))
        train_idxs, test_idxs = train_test_split(
            idxs,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify
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

        data_train = SequentialDataset(np.vstack(X_train), y=y_train, lengths=lengths_train, classes=classes)
        data_test = SequentialDataset(np.vstack(X_test), y=y_test, lengths=lengths_test, classes=classes)

        return data_train, data_test


    def iter_by_class(self) -> Iterator[Tuple[Array, Array, int]]:
        """Subsets the observation sequences by class.

        :raises: ``AttributeError`` - If ``y`` was not provided to :func:`__init__`, or is not categorical.
        :return: Generator iterating over classes, yielding:

            - ``X`` subset of sequences belonging to the class.
            - Lengths corresponding to the ``X`` subset.
            - Class used to subset ``X``.
        """
        if self._y is None:
            raise AttributeError('No `y` values were provided during initialization')

        if self._classes is None:
            raise RuntimeError('Cannot iterate by class on real-valued targets')

        for c in self._classes:
            ind = np.argwhere(self._y == c).flatten()
            X, _ = self[ind]
            lengths = self._lengths[ind]
            yield np.vstack(X), lengths, c


    @staticmethod
    def _get_idxs(lengths):
        ends = lengths.cumsum()
        starts = np.zeros_like(ends)
        starts[1:] = ends[:-1]
        return np.c_[starts, ends]


    @staticmethod
    def _iter_X(X, idxs):
        for start, end in idxs:
            yield X[start:end]


    def __len__(self):
        return len(self._lengths)


    def __getitem__(self, i):
        idxs = np.atleast_2d(self._idxs[i])
        X = [x for x in self._iter_X(self._X, idxs)]
        X = X[0] if isinstance(i, int) and len(X) == 1 else X
        return X if self._y is None else (X, self._y[i])


    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


    @property
    def X(self) -> Array:
        """Observation sequences."""
        return self._X


    @property
    def y(self) -> Array:
        """Outputs corresponding to ``X``.

        :raises: ``AttributeError`` - If ``y`` was not provided to :func:`__init__`.
        """
        if self._y is None:
            raise AttributeError('No `y` values were provided during initialization')
        return self._y


    @property
    def lengths(self) -> Array[int]:
        """Lengths corresponding to ``X``."""
        return self._lengths


    @property
    def classes(self) -> Optional[Array[int]]:
        """Set of unique classes in ``y``. If ``y`` is not categorical, then ``None``."""
        return self._classes


    @property
    def idxs(self) -> Array[int]:
        """Observation sequence start and end indices."""
        return self._idxs


    @property
    def X_y(self) -> Tuple[Array, Array]:
        """Observation sequences and corresponding outputs.

        :raises: ``AttributeError`` - If ``y`` was not provided to :func:`__init__`.
        """
        if self._y is None:
            raise AttributeError('No `y` values were provided during initialization')
        return self._X_y


    @property
    def X_lengths(self) -> Tuple[Array, Array[int]]:
        """Observation sequences and corresponding lengths."""
        return self._X_lengths


    @property
    def X_y_lengths(self) -> Tuple[Array, Array, Array[int]]:
        """Observation sequences and corresponding outputs and lengths.

        :raises: ``AttributeError`` - If ``y`` was not provided to :func:`__init__`.
        """
        if self._y is None:
            raise AttributeError('No `y` values were provided during initialization')
        return self._X_y_lengths


    def save(self, path: Union[str, pathlib.Path, IO], compress: bool = True):
        """Stores the dataset in ``.npz`` format.

        See :func:`numpy:numpy.savez` and :func:`numpy:numpy.savez_compressed`.

        :param path: Location to store the dataset.
        :param compress: Whether or not to compress the dataset.

        See Also
        --------
        load:
            Loads a stored dataset in ``.npz`` format.
        """
        arrs = {
            'X': self._X,
            'lengths': self._lengths
        }

        if self._y is not None:
            arrs['y'] = self._y

        if self._classes is not None:
            arrs['classes'] = self._classes

        save_fun = np.savez_compressed if compress else np.savez
        save_fun(path, **arrs)


    @classmethod
    def load(cls, path: Union[str, pathlib.Path, IO]) -> SequentialDataset:
        """Loads a stored dataset in ``.npz`` format.

        See :func:`numpy:numpy.load`.

        :param path: Location to store the dataset.
        :return: The loaded dataset.

        See Also
        --------
        save:
            Stores the dataset in ``.npz`` format.
        """
        return cls(**np.load(path))


    def copy(self) -> SequentialDataset:
        """Creates a copy of the dataset.

        :return: Dataset copy.
        """
        params = {
            "X": copy.deepcopy(self._X),
            "lengths": copy.deepcopy(self._lengths),
        }

        if self._y is not None:
            params["y"] = copy.deepcopy(self._y)

        if self._classes is not None:
            params["classes"] = copy.deepcopy(self._classes)

        return SequentialDataset(**params)
