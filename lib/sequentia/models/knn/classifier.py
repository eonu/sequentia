from .base import KNNConfig, KNNMixin
from ..base import Classifier
from ...utils import validate_params, requires_fit, override_params

class KNNClassifier(KNNMixin, Classifier):
    @validate_params(using=KNNConfig)
    def __init__(self, *,
        k = 1,
        weighting = 'uniform',
        window = 1,
        independent = False,
        use_c = False,
        n_jobs = 1,
        progress_bar = False, # TODO: Add to Pydantic config and set self.progress_bar = progress_bar
        random_state = None
    ):
        self.k = k
        self.weighting = weighting
        self.window = window
        self.independent = independent
        self.use_c = use_c
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, y, lengths):
        pass

    @requires_fit
    def predict(self, X, lengths):
        pass

    @requires_fit
    def predict_proba(self, X, lengths):
        pass

    @requires_fit
    def predict_scores(self, X, lengths):
        pass

    @validate_params(using=KNNConfig)
    @override_params(KNNConfig.fields(), temporary=False)
    def set_params(self, **kwargs):
        pass
