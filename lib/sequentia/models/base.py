from __future__ import annotations
from typing import Optional, Union, Any

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import accuracy_score, r2_score

from sequentia.utils.validation import Array
from sequentia.utils.decorators import requires_fit

__all__ = ['Classifier', 'Regressor']

class Classifier(BaseEstimator, ClassifierMixin):
    def fit(
        self,
        X: Union[Array[int], Array[float]],
        y: Array[int],
        lengths: Optional[Array[int]] = None
    ) -> Classifier:
        raise NotImplementedError

    def predict(
        self,
        X: Union[Array[int], Array[float]],
        lengths: Optional[Array[int]] = None
    ) -> Array[int]:
        raise NotImplementedError

    def predict_proba(
        self,
        X: Union[Array[int], Array[float]],
        lengths: Optional[Array[int]] = None
    ) -> Array[float]:
        raise NotImplementedError

    def predict_scores(
        self,
        X: Union[Array[int], Array[float]],
        lengths: Optional[Array[int]] = None
    ) -> Array[float]:
        raise NotImplementedError

    @requires_fit
    def score(
        self,
        X: Union[Array[int], Array[float]],
        y: Array[int],
        lengths: Optional[Array[int]] = None,
        sample_weight: Optional[Any] = None
    ) -> float:
        y_pred = self.predict(X, lengths)
        return accuracy_score(y, y_pred, sample_weight=sample_weight)

class Regressor(BaseEstimator, RegressorMixin):
    def fit(
        self,
        X: Array[float],
        y: Array[float],
        lengths: Optional[Array[int]] = None
    ) -> Regressor:
        raise NotImplementedError

    def predict(
        self,
        X: Array[float],
        lengths: Optional[Array[int]] = None
    ) -> Array[float]:
        raise NotImplementedError

    @requires_fit
    def score(
        self,
        X: Array[float],
        y: Array[float],
        lengths: Optional[Array[int]] = None,
        sample_weight: Optional[Any] = None
    ) -> float:
        y_pred = self.predict(X, lengths)
        return r2_score(y, y_pred, sample_weight=sample_weight)
