# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Free Spoken Digit Dataset."""

from __future__ import annotations

import importlib.resources
import operator

import numpy as np
import pydantic as pyd

import sequentia.datasets.data
from sequentia._internal import _data
from sequentia.datasets.base import SequentialDataset

__all__ = ["load_digits"]


@pyd.validate_call
def load_digits(
    *, digits: set[pyd.conint(ge=0, le=9)] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
) -> SequentialDataset:
    """Load a dataset of MFCC features of spoken digit audio samples from the
    Free Spoken Digit Dataset.

    The `Free Spoken Digit Dataset (FSDD) <https://github.com/Jakobovski/free-spoken-digit-dataset>`_
    consists of 3000 recordings of the spoken digits 0-9.

    This version consists of 13 MFCC features of 50 recordings for each digit
    by 6 individual speakers.

    Parameters
    ----------
    digits:
        Subset of digits to include in the dataset.

    Returns
    -------
    SequentialDataset
        A dataset object representing the loaded digits.
    """
    # Load the dataset from compressed numpy file
    path = importlib.resources.files(sequentia.datasets.data)
    data = np.load(path / "digits.npz")

    # Fetch arrays from loaded file
    X, y, lengths = operator.itemgetter("X", "y", "lengths")(data)

    # Create a dataset only with sequences having the specified labels
    idx = np.argwhere(np.isin(y, sorted(digits))).flatten()
    ranges = _data.get_idxs(lengths)[idx]
    return SequentialDataset(
        np.vstack(list(_data.iter_X(X, idxs=ranges))),
        y[idx],
        lengths=lengths[idx],
    )
