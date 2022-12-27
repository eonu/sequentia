from pkg_resources import resource_filename
from typing import Iterable, Tuple
from operator import itemgetter

import numpy as np
from pydantic import conint, validator
from sklearn.preprocessing import LabelEncoder

from sequentia.utils.data import SequentialDataset
from sequentia.utils.validation import _Validator
from sequentia.utils.decorators import _validate_params


class _GeneFamiliesValidator(_Validator):
    families: Iterable[conint(ge=0, le=6)] = list(range(7))

    @validator('families')
    def check_families(cls, value):
        value = list(value)
        if len(set(value)) < len(value):
            raise ValueError('Expected gene families to be unique')
        return value

@_validate_params(using=_GeneFamiliesValidator)
def load_gene_families(
    *,
    families: Iterable[int] = list(range(7))
) -> Tuple[SequentialDataset, LabelEncoder]:
    """Loads human DNA sequences grouped by gene family.

    The `Human DNA Sequences <https://www.kaggle.com/datasets/sooryaprakash12/human-dna-sequences>`_ dataset
    consists of 4380 DNA sequences belonging to 7 gene families.

    This dataset has imbalanced classes, and uses an :class:`sklearn:sklearn.preprocessing.LabelEncoder` to
    encode the original symbols (``A``, ``T``, ``C``, ``G``, ``N``) that form the DNA sequences, into integers.

    The gene families have the following class labels:

    - G protein coupled receptors: ``0``
    - Tyrosine kinase: ``1``
    - Tyrosine phosphatase: ``2``
    - Synthetase: ``3``
    - Synthase: ``4``
    - Ion channel: ``5``
    - Transcription: ``6``

    :param families: Subset of gene families to include in the dataset.

    :return:

        - A dataset object representing the loaded genetic data.
        - Label encoder used to encode the observation symbols into integers.
    """
    # Load the dataset from compressed numpy file
    data = np.load(resource_filename('sequentia', 'datasets/data/gene_families.npz'))

    # Fetch arrays from loaded file
    X, y, lengths = itemgetter('X', 'y', 'lengths')(data)

    # Encode the observation symbols into integers
    enc = LabelEncoder()
    X = np.expand_dims(enc.fit_transform(X.flatten()), axis=-1)

    # Select and create a Dataset only with sequences having the specified labels
    idx = np.argwhere(np.isin(y, families)).flatten()
    ranges = SequentialDataset._get_idxs(lengths)[idx]
    data = SequentialDataset(
        np.vstack([x for x in SequentialDataset._iter_X(X, ranges)]),
        y[idx],
        lengths[idx]
    )

    return data, enc
