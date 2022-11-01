from pkg_resources import resource_filename
from typing import Sequence
from operator import itemgetter

import numpy as np
from pydantic import conint, validator

from sequentia.utils.data import SequentialDataset
from sequentia.utils.validation import _Validator
from sequentia.utils.decorators import _validate_params

class DigitsValidator(_Validator):
    numbers: Sequence[conint(ge=0, le=9)] = list(range(10))

    @validator('numbers')
    def check_numbers(cls, value):
        value = list(value)
        if len(set(value)) < len(value):
            raise ValueError('Expected numbers to be unique')
        return value

@_validate_params(using=DigitsValidator)
def load_digits(
    *,
    numbers: Sequence[int] = list(range(10)), 
) -> SequentialDataset:
    """Loads MFCC features of spoken digit audio samples from the Free Spoken Digit Dataset.

    The `Free Spoken Digit Dataset (FSDD) <https://github.com/Jakobovski/free-spoken-digit-dataset>`_
    consists of 3000 recordings of the spoken digits 0-9.

    The dataset consists of 50 recordings of each digit by 6 individual speakers.

    :param numbers: Subset of digits to include in the dataset.    
    :return: A dataset object representing the loaded digits.
    """
    # Load the dataset from compressed numpy file
    data = np.load(resource_filename('sequentia', 'datasets/data/digits.npz'))

    # Fetch arrays from loaded file
    X, y, lengths = itemgetter('X', 'y', 'lengths')(data)

    # Select and create a Dataset only with sequences having the specified labels
    idx = np.argwhere(np.isin(y, numbers)).flatten()
    ranges = SequentialDataset._get_idxs(lengths)[idx]
    return SequentialDataset(
        np.vstack([x for x in SequentialDataset._iter_X(X, ranges)]),
        y[idx],
        lengths[idx]
    )
