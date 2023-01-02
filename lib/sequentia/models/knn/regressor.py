from __future__ import annotations

from typing import Optional, Union, Callable, Any

import numpy as np
from pydantic import NegativeInt, NonNegativeInt, PositiveInt, confloat
from sklearn.utils import check_random_state

from sequentia.models.knn.base import _KNNMixin, _KNNValidator, _defaults
from sequentia.models.base import _Regressor

from sequentia.utils.decorators import _validate_params, _requires_fit, _override_params
from sequentia.utils.data import SequentialDataset
from sequentia.utils.validation import (
    Array,
    _MultivariateFloatSequenceRegressorValidator
)

__all__ = ['KNNRegressor']


class KNNRegressor(_KNNMixin, _Regressor):
    """A k-nearest neighbor regressor that uses DTW as a distance measure for sequence comparison.

    The regressor computes the output as a distance weighted average of the outputs of the sequences within the DTW k-neighborhood of the sequence being predicted.
    """

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
    ) -> KNNRegressor:
        """Initializes the :class:`.KNNRegressor`.

        :param k: Number of neighbors.

        :param weighting: A callable that specifies how distance weighting should be performed.
            The callable should accept a :class:`numpy:numpy.ndarray` of DTW distances, apply an element-wise weighting transformation
            to the matrix of DTW distances, then return an equally-sized :class:`numpy:numpy.ndarray` of weightings.
            If ``None``, then a uniform weighting of 1 will be applied to all distances.

        :param window: The size of the Sakoe—Chiba band global constrant as a fraction of the length of the shortest of the two sequences being compared.

            - A larger window will give more freedom to the DTW alignment, allowing more deviation but leading to potentially slower computation.
              A window of 1 is equivalent to full DTW computation with no global constraint applied.
            - A smaller window will restrict the DTW alignment, and possibly speed up the DTW computation.
              A window of 0 is equivalent to Euclidean distance.

        :param independent: Whether or not to allow features to be warped independently from each other. See [#dtw_multi]_ for an overview of independent and dependent dynamic time warping.

        :param use_c: Whether or not to use fast pure C compiled functions from `dtaidistance <https://github.com/wannesm/dtaidistance>`__ to perform the DTW computations.

        :param n_jobs: Maximum number of concurrently running workers.

            - If 1, no parallelism is used at all (useful for debugging).
            - If -1, all CPUs are used.
            - If < -1, ``(n_cpus + 1 + n_jobs)`` are used — e.g. ``n_jobs=-2`` uses all but one.

        :param random_state: Seed or :class:`numpy:numpy.random.RandomState` object for reproducible pseudo-randomness.
        """
        #: Number of neighbors.
        self.k = k
        #: A callable that specifies how distance weighting should be performed.
        self.weighting = weighting
        #: The size of the Sakoe—Chiba band global constrant as a fraction of the length of the shortest of the two sequences being compared.
        self.window = window
        #: Whether or not to allow features to be warped independently from each other.
        self.independent = independent
        #: Set of possible class labels.
        self.use_c = use_c
        #: Maximum number of concurrently running workers.
        self.n_jobs = n_jobs
        #: Seed or :class:`numpy:numpy.random.RandomState` object for reproducible pseudo-randomness.
        self.random_state = random_state


    def fit(
        self,
        X: Array[float],
        y: Array[float],
        lengths: Optional[Array[int]] = None
    ) -> KNNRegressor:
        """Fits the regressor to the sequence(s) in ``X``.

        :param X: Univariate or multivariate observation sequence(s).

            - Should be a single 1D or 2D array.
            - Should have length as the 1st dimension and features as the 2nd dimension.
            - Should be a concatenated sequence if multiple sequences are provided,
              with respective sequence lengths being provided in the ``lengths`` argument for decoding the original sequences.

        :param y: Outputs corresponding to sequence(s) provided in ``X``.

        :param lengths: Lengths of the observation sequence(s) provided in ``X``.

            - If ``None``, then ``X`` is assumed to be a single observation sequence.
            - ``len(X)`` should be equal to ``sum(lengths)``.

        :return: The fitted regressor.
        """
        data = _MultivariateFloatSequenceRegressorValidator(X=X, y=y, lengths=lengths)
        self.X_ = data.X
        self.y_ = data.y
        self.lengths_ = data.lengths
        self.idxs_ = SequentialDataset._get_idxs(data.lengths)
        self.random_state_ = check_random_state(self.random_state)
        return self


    @_requires_fit
    def predict(
        self,
        X: Array[float],
        lengths: Optional[Array[int]] = None
    ) -> Array[float]:
        """Predicts outputs for the sequence(s) in ``X``.

        :param X: Univariate or multivariate observation sequence(s).

            - Should be a single 1D or 2D array.
            - Should have length as the 1st dimension and features as the 2nd dimension.
            - Should be a concatenated sequence if multiple sequences are provided,
              with respective sequence lengths being provided in the ``lengths`` argument for decoding the original sequences.

        :param lengths: Lengths of the observation sequence(s) provided in ``X``.

            - If ``None``, then ``X`` is assumed to be a single observation sequence.
            - ``len(X)`` should be equal to ``sum(lengths)``.

        :note: This method requires a trained regressor — see :func:`fit`.

        :return: Output predictions.
        """
        _, k_distances, k_outputs = self.query_neighbors(X, lengths, sort=False)
        k_weightings = self._weighting()(k_distances)
        return (k_outputs * k_weightings).sum(axis=1) / k_weightings.sum(axis=1)


    def fit_predict(
        self,
        X: Array[float],
        y: Array[float],
        lengths: Optional[Array[int]] = None
    ) -> Array[float]:
        """Fits the regressor to the sequence(s) in ``X`` and predicts outputs for ``X``.

        :param X: Univariate or multivariate observation sequence(s).

            - Should be a single 1D or 2D array.
            - Should have length as the 1st dimension and features as the 2nd dimension.
            - Should be a concatenated sequence if multiple sequences are provided,
              with respective sequence lengths being provided in the ``lengths`` argument for decoding the original sequences.

        :param y: Outputs corresponding to sequence(s) provided in ``X``.

        :param lengths: Lengths of the observation sequence(s) provided in ``X``.

            - If ``None``, then ``X`` is assumed to be a single observation sequence.
            - ``len(X)`` should be equal to ``sum(lengths)``.

        :return: Output predictions.
        """
        return super().fit_predict(X, y, lengths)


    @_requires_fit
    def score(
        self,
        X: Array[float],
        y: Array[float],
        lengths: Optional[Array[int]] = None,
        sample_weight: Optional[Any] = None
    ) -> float:
        """Calculates the coefficient of determination (R\ :sup:`2`) for the sequence(s) in ``X``.

        :param X: Univariate or multivariate observation sequence(s).

            - Should be a single 1D or 2D array.
            - Should have length as the 1st dimension and features as the 2nd dimension.
            - Should be a concatenated sequence if multiple sequences are provided,
              with respective sequence lengths being provided in the ``lengths`` argument for decoding the original sequences.

        :param y: Outputs corresponding to the observation sequence(s) in ``X``.

        :param lengths: Lengths of the observation sequence(s) provided in ``X``.

            - If ``None``, then ``X`` is assumed to be a single observation sequence.
            - ``len(X)`` should be equal to ``sum(lengths)``.

        :param sample_weight: See :func:`sklearn:sklearn.metrics.r2_score`.

        :note: This method requires a trained regressor — see :func:`fit`.

        :return: Coefficient of determination.
        """
        return super().score(X, y, lengths, sample_weight)


    @_validate_params(using=_KNNValidator)
    @_override_params(['k', 'weighting', 'window', 'independent', 'use_c', 'n_jobs'], temporary=False)
    def set_params(self, **kwargs):
        return self
