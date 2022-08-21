import copy
import warnings

import numpy as np
from sklearn.model_selection import train_test_split

from sequentia.utils.validation import check_classes, BaseSequenceValidator

__all__ = ['SequentialDataset']

class SequentialDataset:
    def __init__(self, X, y=None, lengths=None, classes=None):
        data = BaseSequenceValidator(X=X, lengths=lengths, y=y)

        self._X = data.X
        self._y = data.y
        self._lengths = data.lengths

        self._classes = None
        if self._y is not None and np.issubdtype(self._y.dtype, np.integer):
            self._classes = check_classes(self._y, classes)

        self._idxs = self._get_idxs(self.lengths)

        self._X_y = (self._X, self._y)
        self._X_lengths = (self._X, self._lengths)
        self._X_y_lengths = (self._X, self._y, self._lengths)

    def split(self, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=False):
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

    def iter_by_class(self):
        if self._y is None:
            raise ValueError('No `y` values were provided during initialization')

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

    def __eq__(self, other):
        if not isinstance(other, SequentialDataset):
            return False

        eq = True
        eq &= np.array_equal(self._X, other._X)
        eq &= np.array_equal(self._lengths, other._lengths)

        if type(self._y) == type(other._y):
            if isinstance(self._y, np.ndarray):
                eq &= np.array_equal(self._y, other._y)
        else:
            return False
        
        if type(self._classes) == type(other._classes):
            if isinstance(self._classes, np.ndarray):
                eq &= np.array_equal(self._classes, other._classes)
        else:
            return False

        return eq

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

    def save(self, path, compress=True):
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
    def load(cls, path):
        return cls(**np.load(path))
