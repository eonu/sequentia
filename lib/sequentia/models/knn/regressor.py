from __future__ import annotations
from typing import Union, Optional, Callable

from pydantic import NegativeInt, PositiveInt, confloat

from sequentia.models.knn.base import KNNValidator, KNNMixin
from sequentia.models.base import Regressor

from sequentia.utils.decorators import validate_params, requires_fit, override_params
from sequentia.utils.data import SequentialDataset
from sequentia.utils.validation import (
    Array,
    MultivariateFloatSequenceRegressorValidator
)

__all__ = ['KNNRegressor']

class KNNRegressor(KNNMixin, Regressor):
    """TODO

    weighting must be a non-negative matrix function!
    """

    @validate_params(using=KNNValidator)
    def __init__(
        self,
        *,
        k: PositiveInt = 1,
        weighting: Optional[Callable] = None,
        window: confloat(ge=0, le=1) = 1,
        independent: bool = False,
        use_c: bool = False,
        n_jobs: Union[NegativeInt, PositiveInt] = 1
    ) -> KNNRegressor:
        self.k = k
        self.weighting = weighting
        self.window = window
        self.independent = independent
        self.use_c = use_c
        self.n_jobs = n_jobs

    def fit(
        self,
        X: Array[float],
        y: Array[int],
        lengths: Optional[Array[int]] = None
    ) -> KNNRegressor:
        data = MultivariateFloatSequenceRegressorValidator(X=X, y=y, lengths=lengths)
        self.X_ = data.X
        self.y_ = data.y
        self.lengths_ = data.lengths
        self.idxs_ = SequentialDataset._get_idxs(data.lengths)
        return self

    @requires_fit
    def predict(
        self,
        X: Array[float],
        lengths: Optional[Array[int]] = None
    ) -> Array[float]:
        _, k_distances, k_outputs = self.query_neighbors(X, lengths, sort=False)
        k_weightings = self._weighting()(k_distances)
        return (k_outputs * k_weightings).sum(axis=1) / k_weightings.sum(axis=1)

    @validate_params(using=KNNValidator)
    @override_params(['k', 'weighting', 'window', 'independent', 'use_c', 'n_jobs'], temporary=False)
    def set_params(self, **kwargs):
        return self
