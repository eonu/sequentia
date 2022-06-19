import torch, numpy as np
from sequentia.datasets import Dataset
from sequentia.classifiers.rnn import collate_fn
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

def test_dataset_to_torch():
    batch_size = 16
    torch_set = dataset.to_torch(transform=(lambda x: -x))
    loader = torch.utils.data.DataLoader(torch_set, collate_fn=collate_fn, batch_size=batch_size, shuffle=False, num_workers=0)
    first_batch_sequences, first_batch_lengths, first_batch_labels = next(iter(loader))
    assert torch.equal(first_batch_sequences, -torch.arange(batch_size)[:, None, None])
    assert torch.equal(first_batch_lengths, torch.ones(batch_size, dtype=int))
    assert torch.equal(first_batch_labels, torch.from_numpy(y[:batch_size]))