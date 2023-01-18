from __future__ import annotations

import types
import joblib
import marshal
import warnings
import pathlib
from types import SimpleNamespace
from typing import Optional, Union, Callable, Tuple, List, Any, IO
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
    k=1,
    weighting=None,
    window=1,
    independent=False,
    use_c=False,
    n_jobs=1,
    random_state=None,
)


class _KNNValidator(_Validator):
    k: PositiveInt = _defaults.k
    weighting: Optional[Callable] = _defaults.weighting
    window: confloat(ge=0, le=1) = _defaults.window
    independent: bool = _defaults.independent
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
    _defaults = _defaults


    @_requires_fit
    @_override_params(['k', 'window', 'independent'])
    @_validate_params(using=_KNNValidator)
    def query_neighbors(
        self,
        X: Array[float],
        lengths: Optional[Array[int]] = None,
        sort: bool = True,
        **kwargs,
    ) -> Tuple[
        Array[int],
        Array[float],
        Array
    ]:
        """Queries the k-nearest training observation sequences to each sequence in ``X``.

        :param X: Univariate or multivariate observation sequence(s).

            - Should be a single 1D or 2D array.
            - Should have length as the 1st dimension and features as the 2nd dimension.
            - Should be a concatenated sequence if multiple sequences are provided,
              with respective sequence lengths being provided in the ``lengths`` argument for decoding the original sequences.

        :param lengths: Lengths of the observation sequence(s) provided in ``X``.

            - If ``None``, then ``X`` is assumed to be a single observation sequence.
            - ``len(X)`` should be equal to ``sum(lengths)``.

        :param sort: Whether to sort the neighbors in order of nearest to furthest.

        :param \*\*kwargs: Model parameters to temporarily override (*for experimentation purposes*).

            - ``k``: See :func:`__init__`.
            - ``window``: See :func:`__init__`.
            - ``independent``: See :func:`__init__`.

        :return: K-nearest neighbors for each sequence in ``X``.

            - Indices of the k-nearest training sequences.
            - DTW distances of the k-nearest training sequences.
            - Corresponding outputs of the k-nearest training sequences.
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
        """Calculates a matrix of DTW distances between the sequences in ``X`` and the training sequences.

        :param X: Univariate or multivariate observation sequence(s).

            - Should be a single 1D or 2D array.
            - Should have length as the 1st dimension and features as the 2nd dimension.
            - Should be a concatenated sequence if multiple sequences are provided,
              with respective sequence lengths being provided in the ``lengths`` argument for decoding the original sequences.

        :param lengths: Lengths of the observation sequence(s) provided in ``X``.

            - If ``None``, then ``X`` is assumed to be a single observation sequence.
            - ``len(X)`` should be equal to ``sum(lengths)``.

        :param \*\*kwargs: Model parameters to temporarily override (*for experimentation purposes*).

            - ``window``: See :func:`__init__`.
            - ``independent``: See :func:`__init__`.

        :note: This method requires a trained classifier — see :func:`fit`.

        :return: DTW distance matrix.
        """
        data = _BaseMultivariateFloatSequenceValidator(X=X, lengths=lengths)

        n_jobs = _effective_n_jobs(self.n_jobs, data.lengths)
        dtw_ = self._dtw()

        row_chunk_idxs = np.array_split(SequentialDataset._get_idxs(data.lengths), n_jobs)
        col_chunk_idxs = np.array_split(self.idxs_, n_jobs)

        return np.vstack(
            Parallel(n_jobs=n_jobs, max_nbytes=None)(
                delayed(self._distance_matrix_row_chunk)(
                    row_idxs, col_chunk_idxs, data.X, n_jobs, dtw_
                ) for row_idxs in row_chunk_idxs
            )
        )


    @_override_params(['window', 'independent'])
    @_validate_params(using=_KNNValidator)
    def dtw(self, A: Array[float], B: Array[float], **kwargs) -> float:
        """Calculates the DTW distance between two univariate or multivariate sequences.

        :param A: The first sequence.

        :param B: The second sequence.

        :param \*\*kwargs: Model parameters to temporarily override (*for experimentation purposes*).

            - ``window``: See :func:`__init__`.
            - ``independent``: See :func:`__init__`.

        :return: DTW distance.
        """
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
        """Calculates the DTW matrix between two sequences and plots the optimal warping path.

        :param a: The first sequence.

        :param b: The second sequence.

        :note: Only supports univariate sequences.

        :param \*\*kwargs: Model parameters to temporarily override (*for experimentation purposes*).

            - ``window``: See :func:`__init__`.

        :return: Plot axes.
        """
        from dtaidistance import dtw_visualisation

        a = _SingleUnivariateFloatSequenceValidator(sequence=a).sequence
        b = _SingleUnivariateFloatSequenceValidator(sequence=b).sequence

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
        """Calculates DTW distances between ``X`` and training sequences, and plots a distance histogram.

        :param X: Univariate or multivariate observation sequence(s).

            - Should be a single 1D or 2D array.
            - Should have length as the 1st dimension and features as the 2nd dimension.
            - Should be a concatenated sequence if multiple sequences are provided,
              with respective sequence lengths being provided in the ``lengths`` argument for decoding the original sequences.

        :param lengths: Lengths of the observation sequence(s) provided in ``X``.

            - If ``None``, then ``X`` is assumed to be a single observation sequence.
            - ``len(X)`` should be equal to ``sum(lengths)``.

        :param ax: Plot axes. If ``None``, new axes are created.

        :param \*\*kwargs: Model parameters to temporarily override (*for experimentation purposes*).

            - ``window``: See :func:`__init__`.
            - ``independent``: See :func:`__init__`.

        :note: This method requires a trained classifier — see :func:`fit`.

        :return: Plot axes.
        """
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
    def plot_weight_histogram(
        self,
        X: Array[float],
        lengths: Optional[Array[int]] = None,
        ax: Optional['matplotlib.axes.Axes'] = None,
        **kwargs
    ) -> 'matplotlib.axes.Axes':
        """Calculates DTW weights between ``X`` and training sequences, and plots a weight histogram.

        :param X: Univariate or multivariate observation sequence(s).

            - Should be a single 1D or 2D array.
            - Should have length as the 1st dimension and features as the 2nd dimension.
            - Should be a concatenated sequence if multiple sequences are provided,
              with respective sequence lengths being provided in the ``lengths`` argument for decoding the original sequences.

        :param lengths: Lengths of the observation sequence(s) provided in ``X``.

            - If ``None``, then ``X`` is assumed to be a single observation sequence.
            - ``len(X)`` should be equal to ``sum(lengths)``.

        :param ax: Plot axes. If ``None``, new axes are created.

        :param \*\*kwargs: Model parameters to temporarily override (*for experimentation purposes*).

            - ``weighting``: See :func:`__init__`.
            - ``window``: See :func:`__init__`.
            - ``independent``: See :func:`__init__`.

        :note: This method requires a trained classifier — see :func:`fit`.

        :return: Plot axes.
        """
        import matplotlib.pyplot as plt

        distances = self.compute_distance_matrix(X, lengths)
        weightings = self._weighting()(distances)

        if ax is None:
            _, ax = plt.subplots()
        ax.hist(weightings.flatten())
        return ax


    def _dtw1d(self, a: Array[float], b: Array[float], window: int) -> float:
        """Computes the DTW distance between two univariate sequences."""
        return dtw.distance(a, b, use_c=self.use_c, window=window)


    def _window(self, A: Array[float], B: Array[float]) -> int:
        """TODO"""
        return int(self.window * min(len(A), len(B)))


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
        return self.weighting if callable(self.weighting) else lambda x: np.ones_like(x)


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
            Parallel(n_jobs=n_jobs, max_nbytes=None)(
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
    def save(self, path: Union[str, pathlib.Path, IO]):
        """Serializes and saves a fitted KNN estimator.

        :param path: Location to save the serialized estimator.

        :note: This method requires a trained classifier — see :func:`fit`.

        See Also
        --------
        load:
            Loads and deserializes a fitted KNN estimator.
        """
        # Fetch main parameters and fitted values
        state = {
            'params': self.get_params(),
            'fitted': {k:v for k, v in self.__dict__.items() if k.endswith('_')}
        }

        # Serialize weighting function
        if self.weighting is None:
            state['params']['weighting'] = self.weighting
        else:
            state['params']['weighting'] = marshal.dumps(
                (self.weighting.__code__, self.weighting.__name__)
            )

        # Serialize model
        joblib.dump(state, path)


    @classmethod
    def load(cls, path: Union[str, pathlib.Path, IO]):
        """Loads and deserializes a fitted KNN estimator.

        :param path: Location to load the serialized estimator from.

        :return: Fitted KNN estimator.

        See Also
        --------
        save:
            Serializes and saves a fitted KNN estimator.
        """
        state = joblib.load(path)

        # Deserialize weighting function
        if state['params']['weighting'] is not None:
            weighting, name = marshal.loads(state['params']['weighting'])
            state['params']['weighting'] = types.FunctionType(weighting, globals(), name)

        # Set main parameters
        model = cls(**state['params'])

        # Set fitted values
        for k, v in state['fitted'].items():
            setattr(model, k, v)

        # Return deserialized model
        return model
