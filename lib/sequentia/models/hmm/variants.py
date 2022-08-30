from __future__ import annotations

import re
import warnings
from enum import Enum, unique
from typing import Optional, Union, Dict, Any
from pydantic import NonNegativeInt, PositiveInt, validator

import numpy as np
import hmmlearn.hmm
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state

from sequentia.utils.decorators import validate_params, requires_fit
from sequentia.utils.validation import (
    Array,
    Validator,
    BaseMultivariateFloatSequenceValidator,
    BaseUnivariateCategoricalSequenceValidator
)
from sequentia.models.hmm.topologies import (
    ErgodicTopology,
    TopologyType,
    TOPOLOGY_MAP
)

__all__ = ['HMM', 'GaussianMixtureHMM', 'MultinomialHMM']

DEFAULT_GMM_PARAMS = 'stmcw'
DEFAULT_MULTINOMIAL_PARAMS = 'ste'

class HMMValidator(Validator):
    n_states: PositiveInt = 2
    topology: Optional[TopologyType] = None
    random_state: Optional[Union[NonNegativeInt, np.random.RandomState]] = None
    hmmlearn_kwargs: Dict[str, Any] = dict()

    _DEFAULT_PARAMS = 'st'
    _UNSETTABLE_HMMLEARN_PARAMS = ['random_state', 'init_params', 'params']
    _STATE_PARAMS = []

    @validator('random_state')
    def check_random_state(cls, value):
        return check_random_state(value)

    @validator('hmmlearn_kwargs')
    def check_hmmlearn_kwargs(cls, value):
        params = value.copy()

        for param in value.keys():
            if param in cls._UNSETTABLE_HMMLEARN_PARAMS:
                if param == 'init_params':
                    if set(params[param]) != set(cls._DEFAULT_PARAMS):
                        params[param] = cls._DEFAULT_PARAMS
                        warnings.warn(
                            f"The `init_params` hmmlearn argument cannot be overridden manually - defaulting to all parameters '{cls._DEFAULT_PARAMS}'.\n"
                            f'Use the set_start_probs(), set_transitions(), {", ".join([f"set_state_{state_param}()" for state_param in cls._STATE_PARAMS])} methods to initialize model parameters.'
                        )
                elif param == 'params':
                    if set(params[param]) != set(cls._DEFAULT_PARAMS):
                        params[param] = cls._DEFAULT_PARAMS
                        warnings.warn(
                            f"The `params` hmmlearn argument cannot be overridden manually - defaulting to all parameters '{cls._DEFAULT_PARAMS}'.\n"
                            'Use the freeze() and unfreeze() methods to specify the learnable model parameters.'
                        )
                else:
                    del params[param]
                    warnings.warn(
                        f'The `{param}` hmmlearn argument cannot be overriden manually - '
                        f'use the {cls.__name__.split("Validator")[0]} constructor to specify this argument.'
                    )

        if 'init_params' not in params:
            params['init_params'] = cls._DEFAULT_PARAMS
            warnings.warn(
                f"No initializable parameters set in hmmlearn `init_params` argument - defaulting to '{cls._DEFAULT_PARAMS}'.\n"
                'If you intend to manually initialize all parameters, use the methods:\n'
                '- set_start_probs()\n'
                '- set_transitions()\n'
                + "\n".join([f'- set_state_{state_param}()' for state_param in cls._STATE_PARAMS])
                + '\n'
            )

        if 'params' not in params:
            params['params'] = cls._DEFAULT_PARAMS
            warnings.warn(
                f"No learnable parameters set in hmmlearn `params` argument - defaulting to '{cls._DEFAULT_PARAMS}'.\n"
                'If you intend to make no parameters learnable, call the freeze() method with no arguments.'
            )

        return params

class HMM(BaseEstimator):
    def __init__(
        self,
        n_states: int,
        topology: Optional[str],
        random_state: Optional[Union[int, np.random.RandomState]],
        hmmlearn_kwargs: Dict[str, Any]
    ):
        if type(self) == HMM:
            raise NotImplementedError(
                f'Abstract class {type(self).__name__} cannot be instantiated - '
                'use the subclassing HMMs defined in the sequentia.models.hmm module'
            )

        self.n_states = n_states
        self.topology = topology
        self.random_state = random_state
        self.hmmlearn_kwargs = hmmlearn_kwargs
        self._skip_init_params = set()
        self._skip_params = set()

    def fit(self, X, lengths):
        raise NotImplementedError

    def n_params(self):
        n_params = 0
        if 's' not in self._skip_params:
            n_params += self.model.startprob_.size
        if 't' not in self._skip_params:
            n_params += self.model.transmat_.size
        return n_params

    def bic(self, X, lengths):
        max_log_likelihood = self.model.score(X, lengths)
        n_params = self.n_params()
        n_seqs = len(lengths)
        return n_params * np.log(n_seqs) - 2 * np.log(max_log_likelihood)

    def aic(self, X, lengths):
        max_log_likelihood = self.model.score(X, lengths)
        n_params = self.n_params()
        return 2 * n_params - 2 * np.log(max_log_likelihood)

    def set_start_probs(self, values='random'):
        error = ValueError("Invalid start probabilities - expected: 'uniform', 'random' or an array of probabilities")
        if isinstance(values, str):
            if values in ('uniform', 'random'):
                self._startprob = values
                self._skip_init_params |= set('s')
            else:
                raise error
        else:
            try:
                self._startprob = np.array(values)
                self._skip_init_params |= set('s')
            except Exception as e:
                raise error from e

    def set_transitions(self, values='random'):
        error = ValueError("Invalid transition matrix - expected: 'uniform', 'random' or an array of probabilities")
        if isinstance(values, str):
            if values in ('uniform', 'random'):
                self._transmat = values
                self._skip_init_params |= set('t')
            else:
                raise error
        else:
            try:
                self._transmat = np.array(values)
                self._skip_init_params |= set('t')
            except Exception as e:
                raise error from e

    def freeze(self, params=None):
        self._skip_params |= set(self._modify_params(params))

    def unfreeze(self, params=None):
        self._skip_params -= set(self._modify_params(params))

    def _modify_params(self, params):
        defaults = self.DEFAULT_PARAMS
        error_msg = f"Expected a string consisting of any combination of {defaults}"
        if isinstance(params, str):
            if bool(re.compile(fr'[^{defaults}]').search(params)):
                raise ValueError(error_msg)
        elif params is None:
            params = defaults
        else:
            raise TypeError(error_msg)
        return params

    def _check_init_params(self):
        topology = self.topology_ or ErgodicTopology(self.n_states, check_random_state(self.random_state))

        if 's' in self._skip_init_params:
            if isinstance(self._startprob, str):
                if self._startprob == 'uniform':
                    self._startprob = topology.uniform_start_probs()
                elif self._startprob == 'random':
                    self._startprob = topology.random_start_probs()
            elif isinstance(self._startprob, np.ndarray):
                self._startprob = topology.check_start_probs(self._startprob)
        else:
            if self.topology_ is not None:
                self.set_start_probs(topology.random_start_probs())

        if 't' in self._skip_init_params:
            if isinstance(self._transmat, str):
                if self._transmat == 'uniform':
                    self._transmat = topology.uniform_transitions()
                elif self._transmat == 'random':
                    self._transmat = topology.random_transitions()
            elif isinstance(self._transmat, np.ndarray):
                if topology is not None:
                    self._transmat = topology.check_transitions(self._transmat)
        else:
            if self.topology_ is not None:
                self.set_transitions(topology.random_transitions())

@unique
class CovarianceType(Enum):
    SPHERICAL = 'spherical'
    DIAGONAL = 'diag'
    FULL = 'full'
    TIED = 'tied'

class GaussianMixtureHMMValidator(HMMValidator):
    n_components: PositiveInt = 1
    covariance_type: CovarianceType = CovarianceType.FULL
    hmmlearn_kwargs: Dict[str, Any] = dict(init_params=DEFAULT_GMM_PARAMS, params=DEFAULT_GMM_PARAMS)

    _DEFAULT_PARAMS = DEFAULT_GMM_PARAMS
    _UNSETTABLE_HMMLEARN_PARAMS = HMMValidator._UNSETTABLE_HMMLEARN_PARAMS + ['n_components', 'n_mix', 'covariance_type']
    _STATE_PARAMS = ['means', 'covariances', 'weights']

class GaussianMixtureHMM(HMM):
    DEFAULT_PARAMS = DEFAULT_GMM_PARAMS

    @validate_params(using=GaussianMixtureHMMValidator)
    def __init__(
        self, *,
        n_states: int = 2,
        n_components: int = 1,
        covariance_type: str = 'full',
        topology: Optional[str] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        hmmlearn_kwargs: dict = dict(
            init_params=DEFAULT_GMM_PARAMS,
            params=DEFAULT_GMM_PARAMS
        )
    ):
        super().__init__(n_states, topology, random_state, hmmlearn_kwargs)
        self.n_components = n_components
        self.covariance_type = covariance_type

    def fit(self, X, lengths=None):
        data = BaseMultivariateFloatSequenceValidator(X=X, lengths=lengths)
        self.random_state_ = check_random_state(self.random_state)
        if self.topology is None:
            self.topology_ = None
        else:
            self.topology_ = TOPOLOGY_MAP[TopologyType(self.topology)](self.n_states, self.random_state_)
        self._check_init_params()

        kwargs = self.hmmlearn_kwargs
        kwargs['init_params'] = ''.join(set(kwargs['init_params']) - self._skip_init_params)
        kwargs['params'] = ''.join(set(kwargs['params']) - self._skip_params)

        self.model = hmmlearn.hmm.GMMHMM(
            n_components=self.n_states,
            n_mix=self.n_components,
            covariance_type=self.covariance_type,
            random_state=self.random_state_,
            **kwargs
        )

        for attr in ('startprob', 'transmat', 'means', 'covars', 'weights'):
            if hasattr(self, f'_{attr}'):
                setattr(self.model, f'{attr}_', getattr(self, f'_{attr}'))

        self.model.fit(data.X, lengths=data.lengths)
        self.n_seqs_ = len(data.lengths)

        return self

    @requires_fit
    def n_params(self):
        n_params = super().n_params()
        if 'm' not in self._skip_params:
            n_params += self.model.means_.size
        if 'c' not in self._skip_params:
            n_params += self.model.covars_.size
        if 'w' not in self._skip_params:
            n_params += self.model.weights_.size
        return n_params

    @requires_fit
    def bic(self, X, lengths=None):
        data = BaseMultivariateFloatSequenceValidator(X=X, lengths=lengths)
        return super().bic(data.X, data.lengths)

    @requires_fit
    def aic(self, X, lengths=None):
        data = BaseMultivariateFloatSequenceValidator(X=X, lengths=lengths)
        return super().aic(data.X, data.lengths)

    def set_state_means(self, values):
        self._means = Array[float].validate_type(values)
        self._skip_init_params |= set('m')

    def set_state_covariances(self, values):
        self._covars = Array[float].validate_type(values)
        self._skip_init_params |= set('c')

    def set_state_weights(self, values):
        self._weights = Array[float].validate_type(values)
        self._skip_init_params |= set('w')

class MultinomialHMMValidator(HMMValidator):
    hmmlearn_kwargs: Dict[str, Any] = dict(init_params=DEFAULT_MULTINOMIAL_PARAMS, params=DEFAULT_MULTINOMIAL_PARAMS)

    _DEFAULT_PARAMS = DEFAULT_MULTINOMIAL_PARAMS
    _STATE_PARAMS = ['emissions']

class MultinomialHMM(HMM):
    DEFAULT_PARAMS = DEFAULT_MULTINOMIAL_PARAMS

    @validate_params(using=MultinomialHMMValidator)
    def __init__(
        self, *,
        n_states: int = 2,
        topology: Optional[str] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        hmmlearn_kwargs: dict = dict(
            init_params=DEFAULT_MULTINOMIAL_PARAMS,
            params=DEFAULT_MULTINOMIAL_PARAMS
        )
    ):
        super().__init__(n_states, topology, random_state, hmmlearn_kwargs)

    def fit(
        self,
        X: Array[int],
        lengths: Optional[Array[int]] = None
    ) -> MultinomialHMM:
        data = BaseUnivariateCategoricalSequenceValidator(X=X, lengths=lengths)
        self.random_state_ = check_random_state(self.random_state)
        if self.topology is None:
            self.topology_ = None
        else:
            self.topology_ = TOPOLOGY_MAP[TopologyType(self.topology)](self.n_states, self.random_state_)
        self._check_init_params()

        kwargs = self.hmmlearn_kwargs
        kwargs['init_params'] = ''.join(set(kwargs['init_params']) - self._skip_init_params)
        kwargs['params'] = ''.join(set(kwargs['params']) - self._skip_params)

        self.model = hmmlearn.hmm.MultinomialHMM(
            n_components=self.n_states,
            random_state=self.random_state_,
            **kwargs
        )

        for attr in ('startprob', 'transmat', 'emissionprob'):
            if hasattr(self, f'_{attr}'):
                setattr(self.model, f'{attr}_', getattr(self, f'_{attr}'))

        self.model.fit(data.X, lengths=data.lengths)
        self.n_seqs_ = len(data.lengths)

        return self

    @requires_fit
    def n_params(self) -> int:
        n_params = super().n_params()
        if 'e' not in self._skip_params:
            n_params += self.model.emissionprob_.size
        return n_params

    @requires_fit
    def bic(
        self,
        X: Array[int],
        lengths: Optional[Array[int]] = None
    ) -> float:
        data = BaseUnivariateCategoricalSequenceValidator(X=X, lengths=lengths)
        return super().bic(data.X, data.lengths)

    @requires_fit
    def aic(
        self,
        X: Array[int],
        lengths: Optional[Array[int]] = None
    ) -> float:
        data = BaseUnivariateCategoricalSequenceValidator(X=X, lengths=lengths)
        return super().aic(data.X, data.lengths)

    def set_state_emissions(self, values):
        self._emissionprob = Array[float].validate_type(values)
        self._skip_init_params |= set('e')
