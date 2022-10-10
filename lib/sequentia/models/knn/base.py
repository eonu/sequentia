from __future__ import annotations

import types
import joblib
import marshal
import warnings
from types import SimpleNamespace
from typing import Optional, Union, Callable, Tuple, List, Any
from joblib import Parallel, delayed

import numpy as np
from pydantic import NegativeInt, NonNegativeInt, PositiveInt, confloat, validator
from dtaidistance import dtw, dtw_ndim
from sklearn.utils import check_random_state

from sequentia.utils.data import SequentialDataset
from sequentia.utils.multiprocessing import _effective_n_jobs
from sequentia.utils.decorators import (
    _validate_params,
    _override_params,
    _requires_fit,
    _check_plotting_dependencies,
)
from sequentia.utils.validation import (
    Array,
    _Validator,
    _BaseMultivariateFloatSequenceValidator,
    _SingleUnivariateFloatSequenceValidator,
    _SingleMultivariateFloatSequenceValidator,
)

dtw_cc = None
try:
    from dtaidistance import dtw_cc
except ImportError:
    pass

_defaults = SimpleNamespace(
    k=5,
    weighting=None,
    window=1,
    independent=False,
    classes=None,
    use_c=False,
    n_jobs=1,
    random_state=None,
)

class _KNNValidator(_Validator):
    k: PositiveInt = _defaults.k
    weighting: Optional[Callable] = _defaults.weighting
    window: confloat(ge=0, le=1) = _defaults.window
    independent: bool = _defaults.independent
    classes: Optional[Array[int]] = _defaults.classes
    use_c: bool = _defaults.use_c
    n_jobs: Union[NegativeInt, PositiveInt] = _defaults.n_jobs
    random_state: Optional[Union[NonNegativeInt, np.random.RandomState]] = _defaults.random_state

    @validator('use_c')
    def check_use_c(cls, value):
        use_c = value
        if use_c and dtw_cc is None:
            warnings.warn('DTAIDistance C library not available - using Python implementation', ImportWarning)
            use_c = False
        return use_c

    @validator('random_state')
    def check_random_state(cls, value):
        return check_random_state(value)

class _KNNMixin:
    @_validate_params(using=_KNNValidator)
    def __init__(
        self,
        *,
        k: PositiveInt = _defaults.k,
        weighting: Optional[Callable] = _defaults.weighting,
        window: confloat(ge=0, le=1) = _defaults.window,
        independent: bool = _defaults.independent,
        use_c: bool = _defaults.use_c,
        n_jobs: Union[NegativeInt, PositiveInt] = _defaults.n_jobs,
        random_state: Optional[Union[NonNegativeInt, np.random.RandomState]] = _defaults.random_state
    ):
        """
        :param k: Number of neighbors.
        :param weighting: A callable that specifies how distance weighting should be performed.
            The callable should accept a :class:`numpy:numpy.ndarray` of DTW distances, apply an element-wise weighting transformation
            to the matrix of DTW distances, then return an equally-sized :class:`numpy:numpy.ndarray` of weightings.
            If ``None``, then a uniform weighting of 1 will be applied to all distances.
        :param window: The width of the Sakoe—Chiba band global constrant as a fraction of the length of the longest of the two sequences being compared.

            - A larger constraint will speed up the DTW alignment by restricting the maximum deviation from the diagonal of the DTW matrix.
            - Too much constraint may lead to poor alignment.

            The default value of 1 corresponds to full DTW computation with no global constraint applied.
        :param independent: Whether or not to allow features to be warped independently from each other. See [#dtw_multi]_ for an overview of independent and dependent dynamic time warping.
        :param use_c: Whether or not to use fast pure C compiled functions from `dtaidistance <https://github.com/wannesm/dtaidistance>`__ to perform the DTW computations.
        :param n_jobs: Number of jobs to run in parallel.

            - Setting this to -1 will use all available CPU cores.
            - Setting this to values below -1 will use ``(n_cpus + 1 + n_jobs)`` CPUs, e.g. ``n_job=-2`` will use all but one CPU.
        :param random_state: Seed or :class:`numpy:numpy.random.RandomState` object for reproducible pseudo-randomness.
        """

        self.k = k
        self.weighting = weighting
        self.window = window
        self.independent = independent
        self.use_c = use_c
        self.n_jobs = n_jobs
        self.random_state = random_state

    @_requires_fit
    @_override_params(['k', 'window', 'independent'])
    @_validate_params(using=_KNNValidator)
    def query_neighbors(
        self,
        X: Array[float],
        lengths: Optional[Array[int]] = None,
        sort: bool = True,
        **kwargs
    ) -> Tuple[
        Array[int],
        Array[float],
        Array
    ]:
        """Queries the k nearest neighbors in the training set for each sequence in X

        TODO
        """

        distances = self.compute_distance_matrix(X, lengths)
        partition_by = range(self.k) if sort else self.k
        k_idxs = np.argpartition(distances, partition_by, axis=1)[:, :self.k]
        k_distances = np.take_along_axis(distances, k_idxs, axis=1)
        k_outputs = self.y_[k_idxs]
        return k_idxs, k_distances, k_outputs

    @_requires_fit
    @_override_params(['window', 'independent'])
    @_validate_params(using=_KNNValidator)
    def compute_distance_matrix(
        self,
        X: Array[float],
        lengths: Optional[Array[int]] = None,
        **kwargs
    ) -> Array[float]:
        """Calculates a matrix of DTW distances between the provided data
        and the training data.

        TODO
        """

        data = _BaseMultivariateFloatSequenceValidator(X=X, lengths=lengths)

        n_jobs = _effective_n_jobs(self.n_jobs, data.lengths)
        dtw_ = self._dtw()

        row_chunk_idxs = np.array_split(SequentialDataset._get_idxs(data.lengths), n_jobs)
        col_chunk_idxs = np.array_split(self.idxs_, n_jobs)

        return np.vstack(
            Parallel(n_jobs=n_jobs)(
                delayed(self._distance_matrix_row_chunk)(
                    row_idxs, col_chunk_idxs, data.X, n_jobs, dtw_
                ) for row_idxs in row_chunk_idxs
            )
        )

    @_override_params(['window', 'independent'])
    @_validate_params(using=_KNNValidator)
    def dtw(self, A: Array[float], B: Array[float], **kwargs) -> float:
        """TODO"""

        A = _SingleMultivariateFloatSequenceValidator(sequence=A).sequence
        B = _SingleMultivariateFloatSequenceValidator(sequence=B).sequence
        return self._dtw(A, B)

    @_check_plotting_dependencies
    @_override_params(['window'])
    @_validate_params(using=_KNNValidator)
    def plot_warping_path_1d(
        self,
        a: Array[float],
        b: Array[float],
        **kwargs
    ) -> 'matplotlib.axes.Axes':
        """TODO"""

        from dtaidistance import dtw_visualisation

        a = _SingleUnivariateFloatSequenceValidator(sequence=a).sequence
        b = _SingleUnivariateFloatSequenceValidator(sequence=b).sequence

        if self.independent:
            warnings.warn('Warping paths cannot be plotted with independent warping - using dependent warping')

        window = self._window(a, b)
        _, paths = dtw.warping_paths(a, b, window=window)
        best_path = dtw.best_path(paths)

        return dtw_visualisation.plot_warpingpaths(a, b, paths, best_path)

    @_check_plotting_dependencies
    @_requires_fit
    @_override_params(['window', 'independent'])
    @_validate_params(using=_KNNValidator)
    def plot_dtw_histogram(
        self,
        X: Array[float],
        lengths: Optional[Array[int]] = None,
        ax: Optional['matplotlib.axes.Axes'] = None,
        **kwargs
    ) -> 'matplotlib.axes.Axes':
        """TODO"""

        import matplotlib.pyplot as plt

        distances = self.compute_distance_matrix(X, lengths)

        if ax is None:
            _, ax = plt.subplots()
        ax.hist(distances.flatten())
        return ax

    @_check_plotting_dependencies
    @_requires_fit
    @_override_params(['weighting', 'window', 'independent'])
    @_validate_params(using=_KNNValidator)
    def plot_score_histogram(
        self,
        X: Array[float],
        lengths: Optional[Array[int]] = None,
        ax: Optional['matplotlib.axes.Axes'] = None,
        **kwargs
    ) -> 'matplotlib.axes.Axes':
        """TODO"""

        import matplotlib.pyplot as plt

        distances = self.compute_distance_matrix(X, lengths)
        scores = self._weighting()(distances)

        if ax is None:
            _, ax = plt.subplots()
        ax.hist(scores.flatten())
        return ax

    def _dtw1d(self, a: Array[float], b: Array[float], window: int) -> float:
        """Computes the DTW distance between two univariate sequences."""

        return dtw.distance(a, b, use_c=self.use_c, window=window)

    def _window(self, A: Array[float], B: Array[float]) -> int:
        """TODO"""

        return max(1, int(self.window * max(len(A), len(B))))

    def _dtwi(self, A: Array[float], B: Array[float]) -> float:
        """Computes the multivariate DTW distance as the sum of the pairwise per-feature DTW distances,
        allowing each feature to be warped independently."""

        window = self._window(A, B)
        return np.sum([self._dtw1d(A[:, i], B[:, i], window) for i in range(A.shape[1])])

    def _dtwd(self, A: Array[float], B: Array[float]) -> float:
        """Computes the multivariate DTW distance so that the warping of the features depends on each other,
        by modifying the local distance measure."""

        window = self._window(A, B)
        return dtw_ndim.distance(A, B, use_c=self.use_c, window=window)

    def _dtw(self) -> Callable:
        """TODO"""

        return self._dtwi if self.independent else self._dtwd

    def _weighting(self) -> Callable:
        """TODO"""

        if callable(self.weighting):
            return self.weighting
        else:
            return lambda x: np.ones_like(x)

    def _distance_matrix_row_chunk(
        self,
        row_idxs: Array[int],
        col_chunk_idxs: List[Array[int]],
        X: Array[float],
        n_jobs: int,
        dist: Callable
    ) -> Array[float]:
        """Calculates a distance sub-matrix for a subset of rows over all columns.

        TODO
        """

        return np.hstack(
            Parallel(n_jobs=n_jobs)(
                delayed(self._distance_matrix_row_col_chunk)(
                    col_idxs, row_idxs, X, dist
                ) for col_idxs in col_chunk_idxs
            )
        )

    def _distance_matrix_row_col_chunk(
        self,
        col_idxs: Array[int],
        row_idxs: Array[int],
        X: Array[float],
        dist: Callable
    ) -> Array[float]:
        """Calculates a distance sub-matrix for a subset of rows and columns.

        TODO
        """

        distances = np.zeros((len(row_idxs), len(col_idxs)))
        for i, x_row in enumerate(SequentialDataset._iter_X(X, row_idxs)):
            for j, x_col in enumerate(SequentialDataset._iter_X(self.X_, col_idxs)):
                distances[i, j] = dist(x_row, x_col)
        return distances

    @_requires_fit
    def save(self, path: Any):
        """TODO"""

        # Fetch main parameters and fitted values
        state = {
            'params': self.get_params(),
            'fitted': {k:v for k, v in self.__dict__.items() if k.endswith('_')}
        }

        # Serialize weighting function
        state['params']['weighting'] = marshal.dumps(
            (self.weighting.__code__, self.weighting.__name__)
        )

        # Serialize model
        joblib.dump(state, path)

    @classmethod
    def load(cls, path):
        """TODO"""

        state = joblib.load(path)

        # Deserialize weighting function
        weighting, name = marshal.loads(state['params']['weighting'])
        state['params']['weighting'] = types.FunctionType(weighting, globals(), name)

        # Set main parameters
        model = cls(**state['params'])

        # Set fitted values
        for k, v in state['fitted'].items():
            setattr(model, k, v)

        # Return deserialized model
        return model
