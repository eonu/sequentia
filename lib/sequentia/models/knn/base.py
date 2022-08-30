import types
import joblib
import marshal
import warnings
from enum import Enum, unique
from typing import Optional, Union, Callable, Literal, Tuple, List
from joblib import Parallel, delayed

import numpy as np
from pydantic import NegativeInt, NonNegativeInt, PositiveInt, confloat, validator
from dtaidistance import dtw, dtw_ndim
from sklearn.utils import check_random_state

dtw_cc = None
try:
    from dtaidistance import dtw_cc
except ImportError:
    pass

from sequentia.utils.data import SequentialDataset
from sequentia.utils.multiprocessing import effective_n_jobs
from sequentia.utils.decorators import (
    validate_params,
    override_params,
    requires_fit,
    check_plotting_dependencies
)
from sequentia.utils.validation import (
    check_is_fitted,
    Array,
    Validator,
    SingleMultivariateFloatSequenceValidator,
    BaseMultivariateFloatSequenceValidator,
    SingleUnivariateFloatSequenceValidator,
    SingleMultivariateFloatSequenceValidator
)

__all__ = ['WeightingType', 'KNNValidator', 'KNNMixin']

@unique
class WeightingType(Enum):
    UNIFORM = 'uniform'

class KNNValidator(Validator):
    k: PositiveInt = 1,
    weighting: Union[Literal[WeightingType.UNIFORM], Callable] = WeightingType.UNIFORM,
    window: confloat(ge=0, le=1) = 1,
    independent: bool = False,
    classes: Optional[Array[int]] = None,
    use_c: bool = False,
    n_jobs: Union[NegativeInt, PositiveInt] = 1,
    random_state: Optional[Union[NonNegativeInt, np.random.RandomState]] = None

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

class KNNMixin:
    @requires_fit
    @override_params(['k', 'window', 'independent'])
    @validate_params(using=KNNValidator)
    def query_neighbors(
        self,
        X: Array[float],
        lengths: Optional[Array[int]] = None,
        sort: bool = True,
        **kwargs
    ) -> Tuple[
        Array[int],
        Array[float],
        Union[Array[int], Array[float]]
    ]:
        """Queries the k nearest neighbors in the training set for each sequence in X"""
        distances = self.compute_distance_matrix(X, lengths)
        partition_by = range(self.k) if sort else self.k
        k_idxs = np.argpartition(distances, partition_by, axis=1)[:, :self.k]
        k_distances = np.take_along_axis(distances, k_idxs, axis=1)
        k_outputs = self.y_[k_idxs]
        return k_idxs, k_distances, k_outputs

    @requires_fit
    @override_params(['window', 'independent'])
    @validate_params(using=KNNValidator)
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
        data = BaseMultivariateFloatSequenceValidator(X=X, lengths=lengths)

        n_jobs = effective_n_jobs(self.n_jobs, data.lengths)
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

    @override_params(['window', 'independent'])
    @validate_params(using=KNNValidator)
    def dtw(self, A: Array[float], B: Array[float], **kwargs) -> float:
        A = SingleMultivariateFloatSequenceValidator(sequence=A).sequence
        B = SingleMultivariateFloatSequenceValidator(sequence=B).sequence
        return self._dtw(A, B)

    @check_plotting_dependencies
    @override_params(['window'])
    @validate_params(using=KNNValidator)
    def plot_warping_path_1d(
        self,
        a: Array[float],
        b: Array[float],
        **kwargs
    ) -> 'matplotlib.axes.Axes':
        import matplotlib.pyplot as plt
        from dtaidistance import dtw_visualisation

        a = SingleUnivariateFloatSequenceValidator(sequence=a).sequence
        b = SingleUnivariateFloatSequenceValidator(sequence=b).sequence

        if self.independent:
            warnings.warn('Warping paths cannot be plotted with independent warping - using dependent warping')

        window = self._window(a, b)
        _, paths = dtw.warping_paths(a, b, window=window)
        best_path = dtw.best_path(paths)

        return dtw_visualisation.plot_warpingpaths(a, b, paths, best_path)

    @check_plotting_dependencies
    @requires_fit
    @override_params(['window', 'independent'])
    @validate_params(using=KNNValidator)
    def plot_dtw_histogram(
        self,
        X: Array[float],
        lengths: Optional[Array[int]] = None,
        ax: Optional['matplotlib.axes.Axes'] = None,
        **kwargs
    ) -> 'matplotlib.axes.Axes':
        import matplotlib.pyplot as plt

        distances = self.compute_distance_matrix(X, lengths)

        if ax is None:
            _, ax = plt.subplots()
        ax.hist(distances.flatten())
        return ax

    @check_plotting_dependencies
    @requires_fit
    @override_params(['weighting', 'window', 'independent'])
    @validate_params(using=KNNValidator)
    def plot_score_histogram(
        self,
        X: Array[float],
        lengths: Optional[Array[int]] = None,
        ax: Optional['matplotlib.axes.Axes'] = None,
        **kwargs
    ) -> 'matplotlib.axes.Axes':
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
        elif WeightingType(self.weighting) == WeightingType.UNIFORM:
            return lambda x: np.ones_like(x)

    def _distance_matrix_row_chunk(
        self,
        row_idxs: Array[int],
        col_chunk_idxs: List[Array[int]],
        X: Array[float],
        n_jobs: int,
        dist: Callable
    ) -> Array[float]:
        """Calculates a distance sub-matrix for a subset of rows over all columns."""
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
        """Calculates a distance sub-matrix for a subset of rows and columns."""
        distances = np.zeros((len(row_idxs), len(col_idxs)))
        for i, x_row in enumerate(SequentialDataset._iter_X(X, row_idxs)):
            for j, x_col in enumerate(SequentialDataset._iter_X(self.X_, col_idxs)):
                distances[i, j] = dist(x_row, x_col)
        return distances

    @requires_fit
    def save(self, path):
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
