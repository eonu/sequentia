from __future__ import annotations

from typing import Optional, Union, Dict, Literal
from joblib import Parallel, delayed

import numpy as np
from pydantic import NegativeInt, PositiveInt, confloat, validator, root_validator
from sklearn.utils.validation import NotFittedError

from sequentia.models.base import _Classifier
from sequentia.models.hmm.variants import HMM
from sequentia.utils.data import SequentialDataset
from sequentia.utils.multiprocessing import _effective_n_jobs
from sequentia.utils.decorators import _validate_params, _override_params, _requires_fit
from sequentia.utils.validation import (
    _check_classes,
    _check_is_fitted,
    Array,
    _Validator,
)

__all__ = ['HMMClassifier']

class _HMMClassifierValidator(_Validator):
    prior: Optional[Union[Literal["frequency"], Dict[int, confloat(ge=0, le=1)]]] = "frequency"
    classes: Optional[Array[int]] = None
    n_jobs: Union[NegativeInt, PositiveInt] = 1

    @validator('prior')
    def check_prior(cls, value):
        if isinstance(value, dict):
            if not np.isclose(sum(value.values()), 1):
                raise ValueError('Prior distribution must sum to one')
        return value

    @root_validator
    def check_prior_keys_with_classes(cls, values):
        if 'prior' in values and 'classes' in values:
            prior, classes = values['prior'], values['classes']
            if isinstance(prior, dict) and classes is not None:
                if set(prior.keys()) != set(classes):
                    raise ValueError(
                        'Provided classes are not consistent with the provided prior distribution - '
                        'ensure that every label in `classes` is present in `prior`'
                    )
        return values

class HMMClassifier(_Classifier):
    """TODO"""

    @_validate_params(using=_HMMClassifierValidator)
    def __init__(
        self,
        *,
        prior: Optional[Union[Literal["frequency"], Dict[int, confloat(ge=0, le=1)]]] = 'frequency',
        classes: Optional[Array[int]] = None,
        n_jobs: Union[NegativeInt, PositiveInt] = 1
    ) -> HMMClassifier:
        self.prior = prior
        self.classes = classes
        self.n_jobs = n_jobs
        self.models = {}

    def add_model(
        self,
        model: HMM,
        label: int
    ):
        """TODO"""

        if not isinstance(model, HMM):
            raise TypeError('Expected `model` argument to be a type of HMM')
        if len(self.models) > 0:
            if type(model) != type(list(self.models.values())[-1]):
                raise TypeError(
                    f'Model of type {type(model).__name__} must be the same as the models already provided '
                    f'to this {type(self).__name__} instance'
                )
        self.models[int(label)] = model

    def add_models(
        self,
        models: Dict[int, HMM]
    ):
        """TODO"""
        if not all(isinstance(model, HMM) for model in models.values()):
            raise TypeError('Expected all provided `models` to be a type of HMM')
        for label, model in models.items():
            self.add_model(model, label)

    def fit(
        self,
        X: Optional[Array] = None,
        y: Optional[Array[int]] = None,
        lengths: Optional[Array[int]] = None
    ) -> HMMClassifier:
        """TODO"""

        if X is None or y is None:
            if len(self.models) == 0:
                raise RuntimeError(
                    f'Fitted models must be provided to this {type(self).__name__} instance if no training data is provided - '
                    'use add_model() to add fitted models to the classifier object'
                )

            for label, model in self.models.items():
                if not _check_is_fitted(model, return_=True):
                    raise NotFittedError(
                        f'The model corresponding to label {label} must be pre-fitted if '
                        f'no training data is provided to this {type(self).__name__} instance'
                    )

            if self.classes is not None:
                # Same logic as _check_classes()
                classes_np = np.array(self.classes).flatten()
                if not np.issubdtype(classes_np.dtype, np.integer):
                    raise TypeError(f'Expected classes to be integers')
                _, idx = np.unique(classes_np, return_index=True)
                self.classes_ = classes_np[np.sort(idx)]
            else:
                # Fetch classes from provided models
                self.classes_ = np.array(list(self.models.keys()))
        else:
            self.classes_ = _check_classes(Array[int].validate_type(y), self.classes)

        # Check that each label has a HMM (and vice versa)
        if set(self.models.keys()) != set(self.classes_):
            raise ValueError(
                'Classes in the dataset are not consistent with the added models - '
                'ensure that every added model corresponds to a class in the dataset'
            )

        if X is not None and y is not None:
            # Iterate through the dataset by class and fit the corresponding model
            data = self._sequence_classifier_validator(X=X, y=y, lengths=lengths)
            dataset = SequentialDataset(data.X, data.y, data.lengths, self.classes_)
            for X_c, lengths_c, c in dataset.iter_by_class():
                self.models[c].fit(X_c, lengths_c)

        # Set class priors
        if self.prior is None:
            self.prior_ = {c:1/len(self.classes_) for c, _ in self.models.items()}
        elif isinstance(self.prior, str):
            if self.prior == "frequency":
                total_seqs = sum(model.n_seqs_ for model in self.models.values())
                self.prior_ = {c:model.n_seqs_/total_seqs for c, model in self.models.items()}
        elif isinstance(self.prior, dict):
            if set(self.prior.keys()) != set(self.classes_):
                raise ValueError(
                    'Classes in the dataset are not consistent with the classes in `prior` - '
                    'ensure that every provided class prior corresponds to a class in the dataset'
                )
            self.prior_ = self.prior

        return self

    @_requires_fit
    def predict(
        self,
        X: Array,
        lengths: Optional[Array] = None
    ) -> Array[int]:
        scores = self.predict_scores(X, lengths)
        max_score_idxs = scores.argmax(axis=1)
        return self.classes_[max_score_idxs]

    @_requires_fit
    def predict_proba(
        self,
        X: Array,
        lengths: Optional[Array] = None
    ) -> Array[float]:
        proba = self.predict_scores(X, lengths)
        proba -= proba.max(axis=1, keepdims=True)
        proba = np.exp(proba)
        proba /= proba.sum(axis=1, keepdims=True)
        return proba

    @_requires_fit
    def predict_scores(
        self,
        X: Array,
        lengths: Optional[Array] = None
    ) -> Array[float]:
        data = self._base_sequence_validator(X=X, lengths=lengths)
        n_jobs = _effective_n_jobs(self.n_jobs, data.lengths)
        chunk_idxs = np.array_split(SequentialDataset._get_idxs(data.lengths), n_jobs)
        return np.concatenate(
            Parallel(n_jobs=n_jobs)(
                delayed(self._compute_scores_chunk)(idxs, data.X)
                for idxs in chunk_idxs
            )
        )

    @_validate_params(using=_HMMClassifierValidator)
    @_override_params(_HMMClassifierValidator.fields(), temporary=False)
    def set_params(self, **kwargs) -> HMMClassifier:
        return self

    def _compute_scores_chunk(self, idxs, X):
        scores = np.zeros((len(idxs), len(self.classes_)))
        for i, x in enumerate(SequentialDataset._iter_X(X, idxs)):
            scores[i] = self._compute_log_posterior(x)
        return scores

    def _compute_log_posterior(self, x):
        log_posterior = np.full(len(self.classes_), -np.inf)
        for i, k in enumerate(self.classes_):
            model = self.models[k]
            log_prior = np.log(self.prior_[k])
            log_likelihood = model._score(x)
            log_posterior[i] = log_prior + log_likelihood
        return log_posterior

    def _base_sequence_validator(self, **kwargs):
        model = self.models[0]
        return model._base_sequence_validator(**kwargs)

    def _sequence_classifier_validator(self, **kwargs):
        model = self.models[0]
        return model._sequence_classifier_validator(**kwargs)
