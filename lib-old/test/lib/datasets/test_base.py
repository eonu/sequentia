import numpy as np
from sequentia.datasets import Dataset
from ...support import assert_all_equal

# Set seed for reproducible randomness
seed = 0
np.random.seed(seed)
rng = np.random.RandomState(seed)

n_sequences = 100
n_classes = 5
classes = range(n_classes)

X = [np.array([[i]]) for i in range(100)]
y = rng.choice(classes, n_sequences)

dataset = Dataset(X=X, y=y, classes=classes, random_state=rng)

def test_dataset_iter_by_class():
    true_partitions = [[x for (x, label) in zip(X, y) if label == c] for c in classes]
    generated_partitions = [sequences for sequences, _ in dataset.iter_by_class()]
    assert_all_equal(true_partitions, generated_partitions)

def test_dataset_split():
    train_set, test_set = dataset.split(split_size=0.2)
    assert len(train_set) == 80
    assert len(test_set) == 20
    assert sorted((*train_set.y, *test_set.y)) == sorted(y)
