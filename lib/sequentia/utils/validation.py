from typing import Optional

import numpy as np
from pydantic import BaseModel, validator, root_validator
from sklearn.utils.multiclass import unique_labels
from sklearn.multiclass import check_classification_targets

class Validator(BaseModel):
    @classmethod
    def fields(cls):
        return list(cls.__fields__.keys())

class TypedArray(np.ndarray):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate_type

    @classmethod
    def validate_type(cls, val):
        return np.array(val, dtype=cls.inner_type)

class ArrayMeta(type):
    def __getitem__(self, t):
        return type('Array', (TypedArray,), {'inner_type': t})

class Array(np.ndarray, metaclass=ArrayMeta):
    pass

class BaseSequenceValidator(Validator):
    X: Array[None]
    lengths: Optional[Array[int]] = None
    y: Optional[Array[None]] = None

    @validator('X')
    def check_X(cls, X):
        X = np.atleast_1d(X)

        dim = X.ndim
        if dim == 1:
            X = X.reshape(-1, 1)
        elif dim > 2:
            raise ValueError('Expected `X` to have a maximum of two dimensions')

        if len(X) == 0:
            raise ValueError(
                'Expected `X` to have at least one observation '
                f'(len(X)=0)'
            )

        return X

    @validator('lengths')
    def check_lengths(cls, value, values):
        X = values.get('X')
        if X is not None:
            len_X = len(X)

            # Treat whole input as one sequence if no lengths given (and try convert to numpy)
            lengths = np.array(len_X if value is None else value).flatten()
            total_lengths = lengths.sum()

            if total_lengths != len_X:
                raise ValueError(
                    'Sum of provided `lengths` does not match the length of `X` '
                    f'(sum(lengths)={total_lengths}, len(X)={len_X})'
                )
        else:
            raise ValueError('Unable to validate `lengths` as it depends on `X`')
        return lengths

    # Needs lengths
    @validator('y')
    def check_y(cls, value, values):
        y = value
        if y is not None:
            lengths = values.get('lengths')
            if lengths is not None:
                y = np.array(y).flatten()

                len_y = len(y)
                n_seqs = len(lengths)

                if len_y != n_seqs:
                    raise ValueError(
                        'Expected `y` to have the same number of elements as `lengths` '
                        f'(len(y)={len_y}, len(lengths)={n_seqs})'
                    )
            else:
                raise ValueError('Unable to validate `y` as it depends on `lengths`')
        return y

class BaseUnivariateCategoricalSequenceValidator(BaseSequenceValidator):
    X: Array[int]
    lengths: Optional[Array[int]] = None
    y: Optional[Array[None]] = None

class UnivariateCategoricalSequenceClassifierValidator(BaseSequenceValidator):
    X: Array[int]
    lengths: Optional[Array[int]] = None
    y: Array[int]

    @root_validator
    def check_X_is_1d(cls, values):
        X = values['X']
        if X.shape[1] > 1:
            raise ValueError('Only univariate categorical sequences are currently supported')
        return values

class BaseMultivariateFloatSequenceValidator(BaseSequenceValidator):
    X: Array[np.float64]
    lengths: Optional[Array[int]] = None
    y: Optional[Array[None]] = None

class MultivariateFloatSequenceClassifierValidator(BaseSequenceValidator):
    X: Array[np.float64]
    lengths: Optional[Array[int]] = None
    y: Array[int]

class MultivariateFloatSequenceRegressorValidator(BaseSequenceValidator):
    X: Array[np.float64]
    lengths: Optional[Array[int]] = None
    y: Array[np.float64]

class SingleUnivariateFloatSequenceValidator(Validator):
    sequence: Array[np.float64]

    @validator('sequence')
    def check_sequence(cls, sequence):
        return sequence.flatten()

class SingleMultivariateFloatSequenceValidator(Validator):
    sequence: Array[np.float64]

    @validator('sequence')
    def check_sequence(cls, sequence):
        sequence = np.atleast_1d(sequence)

        dim = sequence.ndim
        if dim == 1:
            sequence = sequence.reshape(-1, 1)
        elif dim > 2:
            raise ValueError('Expected sequence to have a maximum of two dimensions')

        if len(sequence) == 0:
            raise ValueError(
                'Expected sequence to have at least one observation (zero length)'
            )

        return sequence

def check_classes(y, classes=None):
    check_classification_targets(y)
    unique_y = unique_labels(y)

    classes_ = None
    if classes is None:
        classes_ = unique_y
    else:
        classes_np = np.array(classes).flatten()
        if not np.issubdtype(classes_np, np.integer):
            raise TypeError(f'Expected classes to be integers')

        _, idx = np.unique(classes_np, return_index=True)
        classes_ = classes_np[np.sort(idx)]
        unseen_labels = set(unique_y) - set(classes_np)
        if len(unseen_labels) > 0:
            raise ValueError(f'Encountered label(s) in `y` not present in specified classes - {unseen_labels}')

    return classes_
