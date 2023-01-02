import pytest

import numpy as np

from sklearn.preprocessing import minmax_scale

from sequentia.datasets import load_digits
from sequentia.preprocessing import transforms

from sequentia.utils import SequentialDataset

from ...support.assertions import assert_equal


@pytest.fixture(scope='module')
def random_state(request):
    return np.random.RandomState(1)


@pytest.fixture(scope='module')
def data(random_state):
    data_= load_digits(digits=[0])
    _, subset = data_.split(test_size=0.2, random_state=random_state, stratify=True)
    return subset


def check_filter(x, xt, func, k):
    """Only works for odd k"""
    assert len(x) == len(xt)
    assert_equal(xt[k // 2], func(x[:k], axis=0))


def test_function_transformer(data):
    # create the transform
    transform = transforms.IndependentFunctionTransformer(minmax_scale)
    # check that fit works - should do nothing
    transform.fit(*data.X_lengths)
    # check that fit_transform works - shouldn't do anything on fit, but should transform
    X_fit_transform = transform.fit_transform(*data.X_lengths)
    # check that transform works
    X_transform = transform.transform(*data.X_lengths)
    # check that fit_transform and transform produce the same transformed data
    assert_equal(X_fit_transform, X_transform)
    # check that features of each sequence are independently scaled to [0, 1]
    for xt in SequentialDataset._iter_X(X_transform, data.idxs):
        assert_equal(xt.min(axis=0), np.zeros(xt.shape[1]))
        assert_equal(xt.max(axis=0), np.ones(xt.shape[1]))


@pytest.mark.parametrize("avg", ["mean", "median"])
@pytest.mark.parametrize("k", [3, 5])
def test_filters(data, random_state, avg, k):
    filter_ = getattr(transforms, f"{avg}_filter")
    check_filter_ = lambda x, xt: check_filter(x, xt, getattr(np, avg), k)

    # check that filters are correctly applied for a single sequence
    n_features = 2
    x = random_state.rand(10 * n_features).reshape(-1, n_features)
    xt = filter_(x, k)
    check_filter_(x, xt)

    # create a transform using the filter, passing k
    transform = transforms.IndependentFunctionTransformer(filter_, kw_args={"k": k})
    Xt = transform.transform(data.X, data.lengths)

    # check that filters are correctly applied for multiple sequences
    idxs = SequentialDataset._get_idxs(data.lengths)
    for x, xt in zip(*map(lambda X: SequentialDataset._iter_X(X, idxs), (data.X, Xt))):
        check_filter_(x, xt)
