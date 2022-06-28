from lib2to3.pytree import Base
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

class Classifier(BaseEstimator, ClassifierMixin):
    pass

class Regressor(BaseEstimator, RegressorMixin):
    pass
