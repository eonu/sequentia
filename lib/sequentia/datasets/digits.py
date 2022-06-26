import numpy as np
from pkg_resources import resource_filename
from .base import Dataset
from ..internals import _Validator

def load_digits(numbers=range(10), random_state=None):
    """Load audio samples of spoken digits from the Free Spoken Digit Dataset.

    The `Free Spoken Digit Dataset (FSDD) <https://github.com/Jakobovski/free-spoken-digit-dataset>`_
    consists of 3000 recordings of the spoken digits 0-9.

    The dataset consists of 50 recordings of each digit by 6 individual speakers.

    Parameters
    ----------
    numbers: array-like of int
        Subset of digits to include in the dataset. Defaults to 0-9.

    random_state: numpy.random.RandomState, int, optional
        A random state object or seed for reproducible randomness.

    Returns
    -------
    dataset: :class:`sequentia.datasets.Dataset`
        A dataset object representing the loaded digits.
    """
    random_state = _Validator().is_random_state(random_state)

    # Load the dataset from compressed numpy file
    data = np.load(resource_filename('sequentia', 'datasets/data/digits.npz'))

    # Fetch arrays from loaded file
    sequences, lengths, labels = data['sequences'], data['lengths'], data['labels']

    # Find the starts and ends of each sequence
    starts = np.zeros_like(lengths)
    starts[1:] = np.cumsum(lengths[:-1])
    ends = np.cumsum(lengths)

    # Split the array into sequences
    X = [sequences[start:end] for start, end in zip(starts, ends)]

    # Select and create a Dataset only with sequences having the specified labels
    X_np = np.array(X, dtype=object)
    idx = np.argwhere(np.isin(labels, numbers)).flatten()
    return Dataset(X_np[idx].tolist(), labels[idx], numbers, random_state)