from __future__ import annotations

import re
import warnings
from copy import deepcopy
from types import SimpleNamespace
from typing import Optional, Union, Dict, Any, Literal
from pydantic import NonNegativeInt, PositiveInt, validator

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state

from sequentia.utils.decorators import _requires_fit
from sequentia.utils.validation import Array, _Validator
from sequentia.models.hmm.topologies import _ErgodicTopology

_defaults = SimpleNamespace(
    n_states=5,
    topology="left-right",
    random_state=None,
    hmmlearn_kwargs=dict(
        init_params="st",
        params="st",
    ),
)

class _HMM(BaseEstimator):
    _base_sequence_validator = None
    _single_sequence_validator = None
    _sequence_classifier_validator = None
    _defaults = _defaults
    _unsettable_hmmlearn_kwargs = ["random_state", "init_params", "params"]

    def __init__(
        self,
        n_states: PositiveInt,
        topology: Optional[str],
        random_state: Optional[Union[NonNegativeInt, np.random.RandomState]],
        hmmlearn_kwargs: Dict[str, Any]
    ) -> _HMM:
        if type(self) == _HMM:
            raise NotImplementedError(
                f'Abstract class {type(self).__name__} cannot be instantiated - '
                'use the subclassing HMMs defined in the sequentia.models.hmm module'
            )

        #: Number of states in the Markov chain.
        self.n_states = n_states
        #: Transition topology of the Markov chain — see :ref:`topologies`.
        self.topology = topology
        #: Seed or :class:`numpy:numpy.random.RandomState` object for reproducible pseudo-randomness.
        self.random_state = random_state
        #: Additional key-word arguments provided to the `hmmlearn <https://hmmlearn.readthedocs.io/en/latest/>`__ HMM constructor.
        self.hmmlearn_kwargs = deepcopy(hmmlearn_kwargs)
        #: Underlying HMM object from `hmmlearn <https://hmmlearn.readthedocs.io/en/latest/>`__ — only set after :func:`fit`.
        self.model = None

        self._skip_init_params = set()
        self._skip_params = set()

    def fit(
        self,
        X: Array,
        lengths: Optional[Array[int]] = None
    ) -> _HMM:
        raise NotImplementedError

    @_requires_fit
    def score(
        self,
        x: Array,
    ) -> float:
        data = self._single_sequence_validator(sequence=x)
        return self._score(data.sequence)

    @_requires_fit
    def n_params(self) -> NonNegativeInt:
        """Retrieves the number of trainable parameters.

        :note: This method requires a trained model — see :func:`fit`.

        :return: Number of trainable parameters.
        """
        n_params = 0
        if 's' not in self._skip_params:
            n_params += self.model.startprob_.size
        if 't' not in self._skip_params:
            n_params += self.model.transmat_.size
        return n_params

    @_requires_fit
    def bic(
        self,
        X: Array,
        lengths: Optional[Array[int]] = None
    ) -> float:
        data = self._base_sequence_validator(X=X, lengths=lengths)
        max_log_likelihood = self.model.score(data.X, data.lengths)
        n_params = self.n_params()
        n_seqs = len(lengths)
        return n_params * np.log(n_seqs) - 2 * np.log(max_log_likelihood)

    @_requires_fit
    def aic(
        self,
        X: Array,
        lengths: Optional[Array[int]] = None
    ) -> float:
        data = self._base_sequence_validator(X=X, lengths=lengths)
        max_log_likelihood = self.model.score(data.X, data.lengths)
        n_params = self.n_params()
        return 2 * n_params - 2 * np.log(max_log_likelihood)

    def set_start_probs(
        self,
        values: Union[Array, Literal["uniform", "random"]] = 'random'
    ):
        """Sets the initial state probabilities.

        If this method is **not** called, initial state probabilities are initialized depending on the value of ``topology`` provided to :func:`__init__`.

        - If ``topology`` was set to ``'ergodic'``, ``'left-right'`` or ``'linear'``, then random probabilities will be assigned according to the topology by calling :func:`set_start_probs` with ``value='random'``.
        - If ``topology`` was set to ``None``, then initial state probabilities will be initialized by `hmmlearn <https://hmmlearn.readthedocs.io/en/latest/>`__.

        :param values: Probabilities or probability type to assign as initial state probabilities.

            - If an ``Array``, should be a vector of starting probabilities for each state.
            - If ``'uniform'``, there is an equal probability of starting in any state.
            - If ``'random'``, the vector of initial state probabilities is sampled
              from a Dirichlet distribution with unit concentration parameters.

        :note: If used, this method should normally be called before :func:`fit`.
        """
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

    def set_transitions(
        self,
        values: Union[Array, Literal["uniform", "random"]] = 'random'
    ):
        """Sets the transition probability matrix.

        If this method is **not** called, transition probabilities are initialized depending on the value of ``topology`` provided to :func:`__init__`:

        - If ``topology`` was set to ``'ergodic'``, ``'left-right'`` or ``'linear'``, then random probabilities will be assigned according to the topology by calling :func:`set_transitions` with ``value='random'``.
        - If ``topology`` was set to ``None``, then initial state probabilities will be initialized by `hmmlearn <https://hmmlearn.readthedocs.io/en/latest/>`__.

        :param values: Probabilities or probability type to assign as state transition probabilities.

            - If an ``Array``, should be a matrix of probabilities where each row must some to one
              and represents the probabilities of transitioning out of a state.
            - If ``'uniform'``, for each state there is an equal probability of transitioning to any state permitted by the topology.
            - If ``'random'``, the vector of transition probabilities for each row is sampled from a
              Dirichlet distribution with unit concentration parameters, according to the shape of the topology.

        :note: If used, this method should normally be called before :func:`fit`.
        """
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

    def freeze(
        self,
        params: str,
    ):
        self._skip_params |= set(self._modify_params(params))

    def unfreeze(
        self,
        params: str,
    ):
        self._skip_params -= set(self._modify_params(params))

    def _modify_params(self, params):
        defaults = deepcopy(self._defaults.hmmlearn_kwargs["params"])
        error_msg = f"Expected a string consisting of any combination of {defaults}"
        if isinstance(params, str):
            if bool(re.compile(fr'[^{defaults}]').search(params)):
                raise ValueError(error_msg)
        else:
            raise TypeError(error_msg)
        return params

    def _check_init_params(self):
        topology = self.topology_ or _ErgodicTopology(self.n_states, check_random_state(self.random_state))

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
                self._transmat = topology.check_transitions(self._transmat)
        else:
            if self.topology_ is not None:
                self.set_transitions(topology.random_transitions())

    def _score(self, x: Array) -> float:
        return self.model.score(x)

class _HMMValidator(_Validator):
    n_states: PositiveInt = _defaults.n_states
    topology: Optional[Literal["ergodic", "left-right", "linear"]] = _defaults.topology
    random_state: Optional[Union[NonNegativeInt, np.random.RandomState]] = _defaults.random_state
    hmmlearn_kwargs: Dict[str, Any] = deepcopy(_defaults.hmmlearn_kwargs)

    _class = _HMM

    @validator('random_state')
    def check_random_state(cls, value):
        return check_random_state(value)

    @validator('hmmlearn_kwargs')
    def check_hmmlearn_kwargs(cls, value):
        params = deepcopy(value)

        defaults = deepcopy(cls._class._defaults.hmmlearn_kwargs["params"])
        setter_methods = [f"{func}()" for func in dir(cls._class) if func.startswith("set") and func != "set_params"]

        for param in value.keys():
            if param in cls._class._unsettable_hmmlearn_kwargs:
                if param == 'init_params':
                    if set(params[param]) != set(defaults):
                        params[param] = defaults
                        warnings.warn(
                            f"The `init_params` hmmlearn argument cannot be overridden manually - defaulting to all parameters '{defaults}'. "
                            f'Use the following methods to initialize model parameters: {", ".join(setter_methods)}.'
                        )
                elif param == 'params':
                    if set(params[param]) != set(defaults):
                        params[param] = defaults
                        warnings.warn(
                            f"The `params` hmmlearn argument cannot be overridden manually - defaulting to all parameters '{defaults}'. "
                            'Use the freeze() and unfreeze() methods to specify the learnable model parameters.'
                        )
                else:
                    del params[param]
                    warnings.warn(
                        f'The `{param}` hmmlearn argument cannot be overriden manually - use the {cls._class.__name__} constructor to specify this argument.'
                    )

        if 'init_params' not in params:
            params['init_params'] = defaults
            warnings.warn(
                f"No initializable parameters set in hmmlearn `init_params` argument - defaulting to '{defaults}'. "
                f'If you intend to manually initialize all parameters, use the following methods: {", ".join(setter_methods)}.'
            )

        if 'params' not in params:
            params['params'] = defaults
            warnings.warn(
                f"No learnable parameters set in hmmlearn `params` argument - defaulting to '{defaults}'. "
                'If you intend to make no parameters learnable, use the freeze() method.'
            )

        return params