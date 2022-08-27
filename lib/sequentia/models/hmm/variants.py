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
    Validator,
    BaseMultivariateFloatSequenceValidator
)
from sequentia.models.hmm.topologies import (
    Topology, 
    LeftRightTopology,
    TopologyType, 
    TOPOLOGY_MAP
)

__all__ = ['HMM', 'GaussianMixtureHMM', 'MultinomialHMM']

DEFAULT_PARAMS = 'stmcw'

class HMMValidator(Validator):
    n_states: PositiveInt = 5
    topology: Optional[TopologyType] = TopologyType.LEFT_RIGHT
    random_state: Optional[Union[NonNegativeInt, np.random.RandomState]] = None

    @validator('random_state')
    def check_random_state(cls, value):
        return check_random_state(value)

class HMM(BaseEstimator):
    @validate_params(using=HMMValidator)
    def __init__(
        self,
        n_states: int = 5,
        topology: str = 'left-right'
    ):
        # TODO
        pass

    def fit(self, X, lengths=None):
        raise NotImplementedError

    def n_params(self):
        raise NotImplementedError

@unique
class CovarianceType(Enum):
    SPHERICAL = 'spherical'
    DIAGONAL = 'diag'
    FULL = 'full'
    TIED = 'tied'

class GaussianMixtureHMMValidator(HMMValidator):
    n_components: PositiveInt = 3
    covariance_type: CovarianceType = CovarianceType.FULL
    hmmlearn_kwargs: Dict[str, Any] = dict(init_params=DEFAULT_PARAMS, params=DEFAULT_PARAMS)

    @validator('hmmlearn_kwargs')
    def check_hmmlearn_kwargs(cls, value):
        params = value.copy()
        unsettable = ('n_components', 'n_mix', 'covariance_type', 'random_state', 'init_params', 'params')

        for param in value.keys():
            if param in unsettable:
                if param == 'init_params':
                    if set(params[param]) != set(DEFAULT_PARAMS):
                        params[param] = DEFAULT_PARAMS
                        warnings.warn(
                            f"The `init_params` hmmlearn argument cannot be overridden manually - defaulting to all parameters 'stmcw'.\n"
                            'Use the set_start_probs(), set_transitions(), set_state_means(), set_state_covariances() and set_state_weights() methods to initialize model parameters.'
                        )
                elif param == 'params':
                    if set(params[param]) != set(DEFAULT_PARAMS):
                        params[param] = DEFAULT_PARAMS
                        warnings.warn(
                            f"The `params` hmmlearn argument cannot be overridden manually - defaulting to all parameters 'stmcw'.\n"
                            'Use the freeze() and unfreeze() methods to specify the learnable model parameters.'
                        )
                else:
                    del params[param]
                    warnings.warn(
                        f'The `{param}` hmmlearn argument cannot be overriden manually.\n'
                        'Use the GaussianMixtureHMM constructor to specify this argument.'
                    )     

        if 'init_params' not in params:
            params['init_params'] = 'stmcw'
            warnings.warn(
                "No initializable parameters set in hmmlearn `init_params` argument - defaulting to 'stmcw'.\n"
                'If you intended to manually initialize all parameters, use the methods:\n'
                '- set_start_probs()\n'
                '- set_transitions()\n'
                '- set_state_means()\n'
                '- set_state_covariances()\n'
                '- set_state_weights()\n'
            )

        if 'params' not in params:
            params['params'] = 'stmcw'
            warnings.warn(
                "No learnable parameters set in hmmlearn `params` argument - defaulting to 'stmcw'.\n"
                'If you intended to make no parameters learnable, call the freeze() method with no arguments.'
            )

        return params

class GaussianMixtureHMM(HMM):
    @validate_params(using=GaussianMixtureHMMValidator)
    def __init__(
        self, *,
        n_states: int = 5, 
        n_components: int = 3, 
        covariance_type: str = 'full', 
        topology: str = 'left-right', 
        random_state: Optional[Union[int, np.random.RandomState]] = None, 
        hmmlearn_kwargs: dict = dict(init_params='stmcw', params='stmcw')
    ):
        self.n_states = n_states
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.topology = topology
        self.random_state = random_state
        self.hmmlearn_kwargs = hmmlearn_kwargs
        self._skip_init_params = set()
        self._skip_params = set()

    def fit(self, X, lengths=None):
        data = BaseMultivariateFloatSequenceValidator(X=X, lengths=lengths)
        self.random_state_ = check_random_state(self.random_state)
        if self.topology is None:
            self.topology_ = None
        else:
            self.topology_ = TOPOLOGY_MAP[TopologyType(self.topology)](self.n_states, self.random_state_)
        self._check_init_params()

        self.model = hmmlearn.hmm.GMMHMM(
            n_components=self.n_states,
            n_mix=self.n_components,
            covariance_type=self.covariance_type,
            random_state=self.random_state_,
            init_params=''.join(set(DEFAULT_PARAMS) - self._skip_init_params),
            params=''.join(set(DEFAULT_PARAMS) - self._skip_params)
        )

        for attr in ('startprob', 'transmat', 'means', 'covars', 'weights'):
            if hasattr(self, f'_{attr}'):
                setattr(self.model, f'{attr}_', getattr(self, f'_{attr}'))

        self.model.fit(data.X, lengths=data.lengths)
        self.n_seqs_ = len(data.lengths)

        return self
    
    @requires_fit
    def n_params(self):
        n_params = 0
        if 's' not in self._skip_params:
            n_params += self.model.startprob_.size
        if 't' not in self._skip_params:
            n_params += self.model.transmat_.size
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
        max_log_likelihood = self.model.score(data.X, lengths=data.lengths)
        n_params = self.n_params()
        n_seqs = len(lengths)
        return n_params * np.log(n_seqs) - 2 * np.log(max_log_likelihood)

    @requires_fit
    def aic(self, X, lengths=None):
        data = BaseMultivariateFloatSequenceValidator(X=X, lengths=lengths)
        max_log_likelihood = self.model.score(data.X, lengths=data.lengths)
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

    def set_state_means(self, values):
        if isinstance(values, np.ndarray):
            self._means = values
            self._skip_init_params |= set('m')
        else:
            raise ValueError("Invalid state means - expected an array")

    def set_state_covariances(self, values):
        if isinstance(values, np.ndarray):
            self._covars = values
            self._skip_init_params |= set('c')
        else:
            raise ValueError("Invalid state covariances - expected an array")

    def set_state_weights(self, values):
        if isinstance(values, np.ndarray):
            self._weights = values
            self._skip_init_params |= set('w')
        else:
            raise ValueError("Invalid state mixture weights - expected an array")

    def freeze(self, params=None):
        self._skip_params |= set(self._modify_params(params))

    def unfreeze(self, params=None):
        self._skip_params -= set(self._modify_params(params))

    def _check_init_params(self):
        if 's' in self._skip_init_params:
            if not hasattr(self, '_startprob'):
                warnings.warn(
                    'Starting probabilities were marked as uninitializable but no starting probabilities were given - '
                    'defaulting to random starting probabilities'
                )
                self.set_start_probs(values='random')
            if isinstance(self._startprob, str):
                if self.topology_ is None:
                    raise RuntimeError('Unable to generate starting probabilities as no topology has been provided')
                if self._startprob == 'uniform':
                    self._startprob = self.topology_.uniform_start_probs()
                elif self._startprob == 'random':
                    self._startprob = self.topology_.random_start_probs()
            elif isinstance(self._startprob, np.ndarray):
                if self.topology_ is None:
                    raise RuntimeError('Unable to set starting probabilities as no topology has been provided')
                self._startprob = self.topology_.check_start_probs(self._startprob)
        else:
            if hasattr(self, '_startprob'):
                warnings.warn(
                    'Starting probabilities were marked to be initialized by hmmlearn but a value was set - '
                    'unsetting provided starting probabilities'
                )
                del self._startprob
            else:
                if self.topology_ is not None:
                    self._startprob = self.topology_.random_start_probs()
                    topology = self.topology_.__class__.__name__
                    warnings.warn(
                        'Starting probabilities were marked to be initialized by hmmlearn but a topology was provided - '
                        f'generating random starting probabilities from {topology}.\n'
                        f'- To set starting probabilities according to {topology}, manually call set_start_probs() before fitting.\n'
                        '- To use hmmlearn initialization, set topology=None.'
                    )

        if 't' in self._skip_init_params:
            if not hasattr(self, '_transmat'):
                warnings.warn(
                    'Transition matrix was marked as uninitializable but no transition matrix was given - '
                    'defaulting to random transition matrix'
                )
                self.set_transitions(values='random')
            if isinstance(self._transmat, str):
                if self.topology_ is None:
                    raise RuntimeError('Unable to generate transition matrix as no topology has been provided')
                if self._transmat == 'uniform':
                    self._transmat = self.topology_.uniform_transitions()
                elif self._transmat == 'random':
                    self._transmat = self.topology_.random_transitions()
            elif isinstance(self._transmat, np.ndarray):
                if self.topology_ is None:
                    raise RuntimeError('Unable to set transition matrix as no topology has been provided')
                self._transmat = self.topology_.check_transitions(self._transmat)
        else:
            if hasattr(self, '_transmat'):
                warnings.warn(
                    'Transition matrix was marked to be initialized by hmmlearn but a value was set - '
                    'unsetting provided transition matrix'
                )
                del self._transmat
            else:
                if self.topology_ is not None:
                    self._transmat = self.topology_.random_transitions()
                    topology = self.topology_.__class__.__name__
                    warnings.warn(
                        'Transition matrix was marked to be initialized by hmmlearn but a topology was provided - '
                        f'generating random transition matrix from {topology}.\n'
                        f'- To set a transition matrix according to {topology}, manually call set_transitions() before fitting.\n'
                        '- To use hmmlearn initialization, set topology=None.'
                    )

        if 'm' in self._skip_init_params:
            if not hasattr(self, '_means'):
                warnings.warn(
                    'State means were marked as uninitializable but no state means were given - '
                    'defaulting to hmmlearn initialization'
                )
                self._skip_init_params -= set('m')

        if 'c' in self._skip_init_params:
            if not hasattr(self, '_covars'):
                warnings.warn(
                    'State covariances were marked as uninitializable but no state covariances were given - '
                    'defaulting to hmmlearn initialization'
                )
                self._skip_init_params -= set('c')

        if 'w' in self._skip_init_params:
            if not hasattr(self, '_weights'):
                warnings.warn(
                    'State mixture weights were marked as uninitializable but no state mixture weights were given - '
                    'defaulting to hmmlearn initialization'
                )
                self._skip_init_params -= set('w')

    def _modify_params(self, params):
        error_msg = "Expected a string consisting of any combination of 's', 't', 'm', 'c', 'w'"
        if isinstance(params, str):
            if bool(re.compile(r'[^stmcw]').search(params)):
                raise ValueError(error_msg)
        elif params is None:
            params = 'stmcw'
        else:
            raise TypeError(error_msg)
        return params

class MultinomialHMM(HMM):
    pass
