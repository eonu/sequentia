from enum import Enum, unique
from typing import Optional, Union, Dict
from pydantic import NonNegativeInt, NegativeInt, PositiveInt, confloat, validator

import numpy as np
from sklearn.utils import check_random_state

from sequentia.utils.decorators import validate_params, override_params, requires_fit
from sequentia.utils.validation import (
    check_classes,
    Array,
    Validator,
    MultivariateFloatSequenceClassifierValidator,
    BaseMultivariateFloatSequenceValidator
)
from sequentia.utils.data import SequentialDataset
from sequentia.models.base import Classifier
from sequentia.models.hmm.variants import HMM

__all__ = ['HMMClassifier']

@unique
class PriorType(Enum):
    FREQUENCY = 'frequency'

class HMMClassifierValidator(Validator):
    prior: Optional[Union[PriorType, Dict[int, confloat(ge=0, le=1)]]] = PriorType.FREQUENCY
    classes: Optional[Array[int]] = None,
    n_jobs: Union[NegativeInt, PositiveInt] = 1,
    random_state: Optional[Union[NonNegativeInt, np.random.RandomState]] = None

    # TODO: Check that prior sums to 1 if dict

    @validator('random_state')
    def check_random_state(cls, value):
        return check_random_state(value)

class HMMClassifier(Classifier):
    @validate_params(using=HMMClassifierValidator)
    def __init__(self, *, 
        prior: Optional[Union[str, dict]] = 'frequency', 
        classes: Optional[Array[int]] = None,
        n_jobs: int = 1,
        random_state: Optional[Union[int, np.random.RandomState]] = None
    ):
        self.prior = prior
        self.classes = classes
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.models = {}

    def add_model(self, model: HMM, label: int):
        if not isinstance(model, HMM):
            raise TypeError('Expected `model` argument to be a type of HMM')
        if not isinstance(label, int):
            raise TypeError('Expected `label` to be an integer')
        self.models[label] = model

    def fit(self, X=None, y=None, lengths=None):
        # TODO: If X is None:
        # - Check that all models have been fitted
        # - Check that classes is same as models dict keys
        # - Don't fit HMMs
        # - Override the random_states of the fitted HMMs

        data = MultivariateFloatSequenceClassifierValidator(X=X, y=y, lengths=lengths)
        self.random_state_ = check_random_state(self.random_state)
        self.classes_ = check_classes(y, self.classes)

        # Check that each label has a HMM (and vice versa)
        if set(self.models.keys()) != set(self.classes_):
            raise ValueError(
                'Classes in the dataset are not consistent with the added models - '
                'ensure that every added model corresponds to a class in the dataset'
            )

        # Update the random state of each model to match the classifier-level state
        for model in self.models.values():
            model.random_state = self.random_state_
        
        # Iterate through the dataset by class and fit the corresponding model - TODO: parallelize?
        dataset = SequentialDataset(data.X, data.y, data.lengths, self.classes_)
        for X_c, lengths_c, c in dataset.iter_by_class():
            self.models[c].fit(X_c, lengths_c)

        # Set class priors
        if self.prior is None:
            self.prior_ = {c:1/len(self.classes_) for c, _ in self.models.items()}
        elif isinstance(self.prior, str):
            if PriorType(self.prior) == PriorType.FREQUENCY:
                self.prior_ = {c:model.n_seqs_/len(data.lengths) for c, model in self.models.items()}
        elif isinstance(self.prior, dict):
            if set(self.prior.keys()) != set(self.classes_):
                raise ValueError(
                    'Classes in the dataset are not consistent with the classes in `prior` - '
                    'ensure that every provided class prior corresponds to a class in the dataset'
                )
            self.prior_ = self.prior

    @requires_fit
    def predict(self, X, lengths=None):
        pass

    @requires_fit
    def predict_proba(self, X, lengths=None):
        pass

    @requires_fit
    def predict_scores(self, X, lengths=None):
        pass

    @validate_params(using=HMMClassifierValidator)
    @override_params(HMMClassifierValidator.fields(), temporary=False)
    def set_params(self, **kwargs):
        return self
