from sequentia.datasets import load_digits

def test_load_digits_full():
    numbers = range(10)
    dataset = load_digits(numbers=numbers)
    assert len(dataset.X) == 3000
    assert len(dataset.y) == 3000
    assert all(((x >= -1) & (x <= 1)).all() for x in dataset.X)
    assert set(dataset.y) == set(numbers)
    assert all((dataset.y == c).sum() == 300 for c in numbers)

def test_load_digits_subset():
    numbers = (0, 2, 6, 7)
    dataset = load_digits(numbers=numbers)
    assert len(dataset.X) == 1200
    assert len(dataset.y) == 1200
    assert all(((x >= -1) & (x <= 1)).all() for x in dataset.X)
    assert set(dataset.y) == set(numbers)
    assert all((dataset.y == c).sum() == 300 for c in numbers)