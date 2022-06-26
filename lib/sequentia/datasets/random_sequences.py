import numpy as np, scipy.spatial
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans
from .base import Dataset
from ..internals import _Validator

def load_random_sequences(
    n_sequences, n_features, n_classes, length_range,
    variance_range=(2, 5), lengthscale_range=(0.2, 0.5),
    random_state=None, tslearn_kwargs={}
):
    """Generates random sequences by sampling from a Gaussian process prior and clustering sequences to obtained labels.

    **Generating sequences**

    The GP prior is parameterised by a kernel function :math:`k_\\theta(x, x')`.
    In this case, we use the squared exponential kernel which is parameterised by :math:`\\theta=(\sigma^2, l)`.

    .. math::
        k_\\theta(x, x') = \\sigma^2 \\exp \\left( - \\frac{(x - x')^2}{2l^2} \\right)

    - :math:`\\sigma^2` is the **variance** of the kernel, which controls the height of the generated functions.
    - :math:`l` is the **lengthscale** of the kernel, which controls the distance between the peaks and troughs of the generated functions.

    For :math:`x` values :math:`\mathbf{x}=(0,\\ldots, n-1)` where :math:`n` is the specified length of the generated sequence, we compute a
    kernel matrix where :math:`K_\\theta(\mathbf{x}, \mathbf{x})_{ij}=k_\\theta(x_i, x_j)`.

    We obtain function values over :math:`X` by drawing samples from a multivariate normal distribution.

    .. math::
        \\mathbf{y} \\sim \\mathcal{N} \\big( \mathbf{0}, K_\\theta(\mathbf{x}, \mathbf{x}) \\big)

    Each sequence is drawn from its own independent GP prior, with a variance and lengthscale sampled from uniform hyperpriors.

    .. math::
        \\sigma^2 \\sim U(a_{\\sigma^2},b_{\\sigma^2}) \qquad l \\sim U(a_l,b_l)

    Where :math:`(a_{\\sigma^2},b_{\\sigma^2})` and :math:`(a_l,b_l)` are given by ``variance_range`` and ``lengthscale_range``.

    If ``n_features`` is more than one, then the same :math:`\\sigma^2` and :math:`l` values are shared along each feature of a single
    sequence, which means that features are often very highly correlated.

    **Generating labels**

    In order to generate labels that are assigned to generated sequences that are somewhat similar,
    we perform clustering over these sequences using :class:`tslearn.clustering.TimeSeriesKMeans`
    and use the resulting cluster labels as the label for each sequence.

    Parameters
    ----------
    n_sequences: int
        Number of sequences to generate.

    n_features: int
        Number of features in each sequences.

    n_classes: int
        Number of classes.

    length_range: tuple(int, int) or int
        Range of values to uniformly sample sequence lengths from.

        If a single value is specified, then all sequences will have equal length.

    variance_range: tuple(float, float) or float
        Lower and upper range of the uniform hyperprior on the GP prior's variance kernel hyperparameter.

        If a single value is specified, then this value is used as the variance rather than placing a hyperprior.

    lengthscale_range: tuple(float, float) or float
        Lower and upper range of the uniform hyperprior on the GP prior's lengthscale kernel hyperparameter.

        If a single value is specified, then this value is used as the lengthscale rather than placing a hyperprior.

    random_state: numpy.random.RandomState, int, optional
        A random state object or seed for reproducible randomness.

    tslearn_kwargs: dict
        Additional key-word arguments for the :class:`tslearn.clustering.TimeSeriesKMeans` constructor.

        The defaults are:

        - ``'metric'``: ``'dtw'``
        - ``'max_iter'``: ``5``
        - ``'max_iter_barycenter'``: ``5``
        - ``'n_jobs'``: ``-1``

        ``'n_clusters'`` and ``'random_state'`` are overwritten by the ``n_classes`` and ``random_state`` arguments
        given to this function.

    Returns
    -------
    dataset: :class:`sequentia.datasets.Dataset`
        A dataset object representing the loaded digits.
    """
    val = _Validator()
    random_state = val.is_random_state(random_state)
    n_classes = val.is_restricted_integer(
        n_classes, lambda x: x <= n_sequences,
        desc='number of classes', expected='no more than n_sequences'
    )

    # Set default tslearn key-word arguments
    tslearn_kwargs['metric'] = tslearn_kwargs.get('metric', 'dtw')
    tslearn_kwargs['max_iter'] = tslearn_kwargs.get('max_iter', 5)
    tslearn_kwargs['max_iter_barycenter'] = tslearn_kwargs.get('max_iter_barycenter', 5)
    tslearn_kwargs['n_jobs'] = tslearn_kwargs.get('n_jobs', -1)

    # Override tslearn key-word arguments
    tslearn_kwargs['n_clusters'] = n_classes
    tslearn_kwargs['random_state'] = random_state

    # Sample sequences from GP priors
    X = []
    for i in range(n_sequences):
        length = _sample_from_range(length_range, int, random_state=random_state)
        variance = _sample_from_range(variance_range, float, random_state=random_state)
        lengthscale = _sample_from_range(lengthscale_range, float, random_state=random_state)
        k = lambda x1, x2: _K(x1, x2, variance=variance, lengthscale=lengthscale)
        X.append(_sample_prior(k, length, n_features, random_state))

    if n_sequences == 1:
        y = np.array([0])
    elif n_classes == 1:
        y = np.zeros(n_sequences)
    else:
        # Cluster sequences to obtain labels
        y = TimeSeriesKMeans(**tslearn_kwargs).fit_predict(to_time_series_dataset(X))

    return Dataset(X, y, range(n_classes), random_state)

def _sample_prior(k, length, n_features, random_state):
    X = np.expand_dims(np.linspace(0, length, length), axis=1)
    return random_state.multivariate_normal(np.zeros(length), k(X, X), size=n_features).T

def _K(x1, x2, variance, lengthscale):
    return variance * np.exp(-scipy.spatial.distance.cdist(x1, x2, 'sqeuclidean') / 2*lengthscale**2)

def _sample_from_range(range_var, var_type, random_state):
    if isinstance(range_var, (int, float)):
        return range_var
    if var_type == int:
        return random_state.choice(range(range_var[0], range_var[1] + 1))
    elif var_type == float:
        return random_state.uniform(*range_var)
