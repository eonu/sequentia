import copy
from typing import Optional

import numpy as np
from pydantic import validator

from .validation import check_classes, BaseSequenceValidator

def iter_X(X, idxs):
    for start, end in idxs:
        yield X[start:end]

# def iter_X(X, lengths):
#     ends = lengths.cumsum()
#     starts = np.zeros_like(ends)
#     starts[1:] = ends[:-1]
#     for start, end in zip(starts, ends):
#         yield X[start:end]

# def iter_X_y(X, y, lengths):
#     for i, x in enumerate(iter_X(X, lengths)):
#         yield x, y[i]

# def iter_X_by_label(X, y, lengths, label):
#     for x, y_ in iter_X_y(X, y, lengths):
#         if y_ == label:
#             yield x

class Dataset:
    # TODO

    def __init__(self, X, y=None, lengths=None, classes=None):
        data = BaseSequenceValidator(X=X, lengths=lengths, y=y)

        self._X = data.X
        self._y = data.y
        self._lengths = data.lengths

        if self._y is not None:
            if np.issubdtype(self._y.dtype, np.integer):
                self._classes = check_classes(self._y, classes)
            else:
                self._classes = None
        else:
            self._classes = None

        self._idxs = self._get_idxs(self.lengths)

        self._X_y = (self._X, self._y)
        self._X_lengths = (self._X, self._lengths)
        self._X_y_lengths = (self._X, self._y, self._lengths)

    @classmethod
    def create(cls, X, y=None):
        # 3D numpy array (multivariate sequences, equal length)
        # 2D numpy array (BxTx1 univariate sequences of equal length)
        # List of lists (univariate sequences)
        # List of lists of lists (multivariate sequences)
        # List of numpy arrays / numpy ragged

        # Try to convert to numpy, catch error, check if dtype=object

        # Check if iterable
        # Check if elements are numbers or more iterables
        #   If numbers, then dealing with a single sequence
        #     Check that y is also length 1 (if y is not None)
        #   If iterables, check if elements are numbers or more iterables
        #     If numbers, we have a ragged array of lists, or maybe all equal
        #       Doesn't matter which case, just find lengths and concatenate
        #     If iterables, we have multivariate sequnces
        #       Use lengths to find out which axis is dimension and which is time step
        #       If all lengths the same along each axis, assume BxTxD

        pass

    def __len__(self):
        return len(self._lengths)

    def __getitem__(self, i):
        idxs = np.atleast_2d(self._idxs[i])
        X = [self._X[start:end] for start, end in idxs]
        X = X[0] if len(X) == 1 else X
        return X if self._y is None else (X, self._y[i])

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def _get_idxs(self, lengths):
        ends = lengths.cumsum()
        starts = np.zeros_like(ends)
        starts[1:] = ends[:-1]
        return np.c_[starts, ends]

    def _iter_by_class(self):
        if self._y is None:
            raise ValueError('No `y` values were provided during initialization')

        if self._classes is None:
            raise RuntimeError('Cannot iterate by class on regression targets')

        for c in self._classes:
            ind = np.argwhere(self._y == c).flatten()
            X, _ = self[ind]
            lengths = self._lengths[ind]
            yield np.vstack(X), lengths, c

    @property
    def X(self):
        return copy.deepcopy(self._X)

    @property
    def y(self):
        if self._y is None:
            raise ValueError('No `y` values were provided during initialization')
        return copy.deepcopy(self._y)

    @property
    def lengths(self):
        return copy.deepcopy(self._lengths)

    @property
    def classes(self):
        return copy.deepcopy(self._classes)

    @property
    def idxs(self):
        return copy.deepcopy(self._idxs)

    @property
    def X_y(self):
        if self._y is None:
            raise ValueError('No `y` values were provided during initialization')
        return copy.deepcopy(self._X_y)

    @property
    def X_lengths(self):
        return copy.deepcopy(self._X_lengths)

    @property
    def X_y_lengths(self):
        if self._y is None:
            raise ValueError('No `y` values were provided during initialization')
        return copy.deepcopy(self._X_y_lengths)
