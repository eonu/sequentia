from __future__ import annotations

from typing import Optional

from sklearn.utils import check_random_state

from sequentia.models.knn.base import _KNNMixin, _KNNValidator
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

    The regressor computes the output as a distance weighted sum of the outputs of the sequences within the DTW k-neighborhood of the sequence being predicted.
    """

    def fit(
        self,
        X: Array[float],
        y: Array[int],
        lengths: Optional[Array[int]] = None
    ) -> KNNRegressor:
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
        _, k_distances, k_outputs = self.query_neighbors(X, lengths, sort=False)
        k_weightings = self._weighting()(k_distances)
        return (k_outputs * k_weightings).sum(axis=1) / k_weightings.sum(axis=1)

    @_validate_params(using=_KNNValidator)
    @_override_params(['k', 'weighting', 'window', 'independent', 'use_c', 'n_jobs'], temporary=False)
    def set_params(self, **kwargs):
        return self
