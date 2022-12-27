import pytest

from sequentia.datasets import load_gene_families

from ...support.assertions import assert_equal

counts = {
    0: 531,
    1: 534,
    2: 349,
    3: 672,
    4: 711,
    5: 240,
    6: 1343
}


@pytest.mark.parametrize('families', [list(range(7)), [2, 5]])
def test_gene_families(families):
    data, enc = load_gene_families(families=families)

    assert set(enc.classes_) == {'A', 'C', 'G', 'N', 'T'}

    assert_equal(data.classes, families)
    assert set(data.y) == set(families)

    for family in families:
        assert (data.y == family).sum() == counts[family]
