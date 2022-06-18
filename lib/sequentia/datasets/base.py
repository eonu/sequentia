import numpy as np
from sklearn.model_selection import train_test_split
from ..internals.versions import is_torch_installed

class Dataset:
    """Represents a generic dataset.

    Parameters
    ----------
    X: array-like
        Data instances.

    y: array-like
        Labels corresponding to data instances.

    classes: array-like
        The complete set of possible classes/labels.

    random_state: numpy.random.RandomState, int, optional
        A random state object or seed for reproducible randomness.
    """
    def __init__(self, X, y, classes, random_state=None):
        self.X = X
        self.y = y
        self.classes = classes
        self.random_state = random_state

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def data(self):
        """Fetch the instances and labels.

        Returns
        -------
        X: array-like
            Data instances.

        y: array-like
            Labels corresponding to data instances.
        """
        return self.X, self.y

    def iter_by_class(self):
        """Generator for iterating through instances partitioned by class.

        Returns
        -------
        instances: generator yielding ``(instances, class)``
            Instances belong to each class.
        """
        X_np = np.array(self.X, dtype=object)
        for c in self.classes:
            yield X_np[self.y == c].tolist(), c

    def split(self, split_size, stratify=True, shuffle=True):
        """Splits the dataset into two smaller :class:`Dataset` objects.

        Parameters
        ----------
        split_size: 0 < float < 1
            Proportion of instances to be allocated to the second split.

        stratify: bool
            Whether or not stratify the split on the labels such that each split
            has a similar distribution of labels.

        shuffle: bool
            Whether or not to shuffle the data before partitioniing it.

        Returns
        -------
        split_1: :class:`Dataset`
            First dataset split.

        split_2: :class:`Dataset`
            Second dataset split.
        """
        X1, X2, y1, y2 = train_test_split(
            self.X, self.y,
            test_size=split_size,
            random_state=self.random_state,
            shuffle=shuffle,
            stratify=(self.y if stratify else None)
        )
        return (
            Dataset(X1, y1, self.classes, self.random_state),
            Dataset(X2, y2, self.classes, self.random_state)
        )

    def to_torch(self, transform=None):
        """Converts the dataset into a :class:`TorchDataset`.

        .. warning::
            This requires a working installation of ``torch``.

        Parameters
        ----------
        transform: callable
            Transformation to apply to each instance.

        Returns
        -------
        torch_dataset: :class:`TorchDataset`
            Torch-compatible dataset.
        """
        if is_torch_installed(silent=False):
            return TorchDataset(self, transform)

# Check that at least the minimum torch version is installed
if is_torch_installed(silent=True):
    import torch.utils

    class TorchDataset(torch.utils.data.Dataset, Dataset):
        """A Torch-compatible dataset subclass of :class:`torch:torch.utils.data.Dataset`.

        .. warning::
            This requires a working installation of ``torch``.

        Parameters
        ----------
        transform: callable
            Transformation to apply to each instance.
        """
        def __init__(self, dataset, transform):
            self.X = dataset.X
            self.y = dataset.y
            self.classes = dataset.classes
            self.transform = transform

        def __getitem__(self, i):
            X, y = torch.from_numpy(self.X[i]), self.y[i]

            # Transform the data if a transformation is provided
            if self.transform is not None:
                X = self.transform(X)

            return X, y