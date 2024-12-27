# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Gene families dataset."""

from __future__ import annotations

import importlib.resources
import operator

import numpy as np
import pydantic as pyd
from sklearn.preprocessing import LabelEncoder

import sequentia.datasets.data
from sequentia._internal import _data
from sequentia.datasets.base import SequentialDataset

__all__ = ["load_gene_families"]


@pyd.validate_call
def load_gene_families(
    *, families: set[pyd.conint(ge=0, le=6)] = {0, 1, 2, 3, 4, 5, 6}
) -> tuple[SequentialDataset, LabelEncoder]:
    """Load a dataset of human DNA sequences grouped by gene family.

    The `Human DNA Sequences <https://www.kaggle.com/datasets/sooryaprakash12/human-dna-sequences>`_
    dataset consists of 4380 DNA sequences belonging to 7 gene families.

    This dataset has imbalanced classes, and uses an
    :class:`sklearn:sklearn.preprocessing.LabelEncoder` to encode the
    original symbols (``A``, ``T``, ``C``, ``G``, ``N``) that form the DNA
    sequences, into integers.

    The gene families have the following class labels:

    - G protein coupled receptors: ``0``
    - Tyrosine kinase: ``1``
    - Tyrosine phosphatase: ``2``
    - Synthetase: ``3``
    - Synthase: ``4``
    - Ion channel: ``5``
    - Transcription: ``6``

    Parameters
    ----------
    families:
        Subset of gene families to include in the dataset.

    Returns
    -------
    tuple[SequentialDataset, sklearn.preprocessing.LabelEncoder]
        - A dataset object representing the loaded genetic data.
        - Label encoder used to encode the observation symbols into integers.
    """
    # Load the dataset from compressed numpy file
    path = importlib.resources.files(sequentia.datasets.data)
    data = np.load(path / "gene_families.npz")

    # Fetch arrays from loaded file
    X, y, lengths = operator.itemgetter("X", "y", "lengths")(data)

    # Encode the observation symbols into integers
    enc = LabelEncoder()
    X = np.expand_dims(enc.fit_transform(X.flatten()), axis=-1)

    # Create a dataset only with sequences having the specified labels
    idx = np.argwhere(np.isin(y, sorted(families))).flatten()
    ranges = _data.get_idxs(lengths)[idx]
    data = SequentialDataset(
        np.vstack(list(_data.iter_X(X, idxs=ranges))),
        y[idx],
        lengths=lengths[idx],
    )

    return data, enc
