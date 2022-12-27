import pytest

from sequentia.datasets import load_digits

from ...support.assertions import assert_equal


@pytest.mark.parametrize('digits', [list(range(10)), [2, 5]])
def test_digits(digits):
    data = load_digits(digits=digits)

    assert len(data) == 300 * len(digits)
    assert_equal(data.classes, digits)
    assert set(data.y) == set(digits)

    for _, lengths, c in data.iter_by_class():
        assert len(lengths) == 300
