# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""Wrapper for a generic hidden Markov Model variant."""

from __future__ import annotations

import abc
import copy
import re
import typing as t
import warnings

import hmmlearn.base
import numpy as np
import pydantic as pyd
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state

from sequentia import enums
from sequentia._internal import _hmm, _validation
from sequentia._internal._typing import Array, FloatArray, IntArray

__all__ = ["BaseHMM"]


class BaseHMM(BaseEstimator, metaclass=abc.ABCMeta):
    """Wrapper for a generic hidden Markov Model variant."""

    _DTYPE: type
    _UNIVARIATE: bool

    @abc.abstractmethod
    def __init__(
        self: BaseHMM,
        *,
        n_states: pyd.PositiveInt,
        topology: enums.TopologyMode | None,
        random_state: pyd.NonNegativeInt | np.random.RandomState | None,
        hmmlearn_kwargs: dict[str, t.Any] | None,
    ) -> BaseHMM:
        self.n_states: int = n_states
        """Number of states in the Markov chain."""

        self.topology: enums.TopologyMode = topology
        """Transition topology of the Markov chain — see :ref:`topologies`."""

        self.random_state: int | np.random.RandomState | None = random_state
        """Seed or :class:`numpy:numpy.random.RandomState` object for
        reproducible pseudo-randomness."""

        self.hmmlearn_kwargs: dict[str, t.Any] = self._check_hmmlearn_kwargs(
            hmmlearn_kwargs
        )
        """Additional key-word arguments provided to the
        `hmmlearn <https://hmmlearn.readthedocs.io/en/latest/>`__ HMM
        constructor."""

        self.model: hmmlearn.base.BaseHMM = None
        """Underlying HMM object from
        `hmmlearn <https://hmmlearn.readthedocs.io/en/latest/>`__ — only set
        after :func:`fit`."""

        self._skip_init_params = set()
        self._skip_params = set()

    def fit(
        self: BaseHMM,
        X: Array,
        *,
        lengths: IntArray | None = None,
    ) -> BaseHMM:
        """Fit the HMM to the sequences in ``X``, using the Baum—Welch
        algorithm.

        Parameters
        ----------
        self: BaseHMM

        X:
            Sequence(s).

        lengths:
            Lengths of the sequence(s) provided in ``X``.

            - If ``None``, then ``X`` is assumed to be a single sequence.
            - ``len(X)`` should be equal to ``sum(lengths)``.

        Returns
        -------
        BaseHMM
            The fitted HMM.
        """
        X, lengths = _validation.check_X_lengths(
            X, lengths=lengths, dtype=self._DTYPE, univariate=self._UNIVARIATE
        )
        self.random_state_ = _validation.check_random_state(self.random_state)
        if self.topology is None:
            self.topology_ = None
        else:
            self.topology_ = _hmm.topologies.TOPOLOGY_MAP[self.topology](
                n_states=self.n_states,
                random_state=self.random_state_,
            )
        self._check_init_params()

        kwargs = copy.deepcopy(self.hmmlearn_kwargs)
        kwargs["init_params"] = "".join(
            set(kwargs["init_params"]) - self._skip_init_params
        )
        kwargs["params"] = "".join(set(kwargs["params"]) - self._skip_params)
        self.model = self._init_hmm(**kwargs)

        for attr in self._hmmlearn_params():
            if hasattr(self, f"_{attr}"):
                setattr(self.model, f"{attr}_", getattr(self, f"_{attr}"))

        self.model.fit(X, lengths=lengths)
        self.n_seqs_ = len(lengths)

        return self

    @_validation.requires_fit
    def score(self: BaseHMM, x: Array, /) -> float:
        """Calculate the log-likelihood of the HMM generating a single
        observation sequence.

        Parameters
        ----------
        self: BaseHMM

        x:
            Sequence.

        Returns
        -------
        float:
            The log-likelihood.

        Notes
        -----
        This method requires a trained model — see :func:`fit`.
        """
        x = _validation.check_X(
            x,
            dtype=self._DTYPE,
            univariate=self._UNIVARIATE,
        )
        return self.model.score(x)

    @abc.abstractproperty
    @_validation.requires_fit
    def n_params(self: BaseHMM) -> int:
        """Number of trainable parameters — requires :func:`fit`."""
        n_params = 0
        if "s" not in self._skip_params:
            n_params += self.model.startprob_.size
        if "t" not in self._skip_params:
            n_params += self.model.transmat_.size
        return n_params

    @_validation.requires_fit
    def bic(
        self: BaseHMM,
        X: Array,
        *,
        lengths: IntArray | None = None,
    ) -> float:
        """The Bayesian information criterion of the model, evaluated with
        the maximum likelihood of ``X``.

        Parameters
        ----------
        self: BaseHMM

        X:
            Sequence(s).

        lengths:
            Lengths of the sequence(s) provided in ``X``.

            - If ``None``, then ``X`` is assumed to be a single sequence.
            - ``len(X)`` should be equal to ``sum(lengths)``.

        Returns
        -------
        float:
            The Bayesian information criterion.

        Notes
        -----
        This method requires a trained model — see :func:`fit`.
        """
        max_log_likelihood = self.model.score(X, lengths=lengths)
        n_params = self.n_params
        n_seqs = len(lengths)
        return n_params * np.log(n_seqs) - 2 * np.log(max_log_likelihood)

    @_validation.requires_fit
    def aic(
        self: BaseHMM,
        X: Array,
        *,
        lengths: IntArray | None = None,
    ) -> float:
        """The Akaike information criterion of the model, evaluated with the
        maximum likelihood of ``X``.

        Parameters
        ----------
        self: BaseHMM

        X:
            Sequence(s).

        lengths:
            Lengths of the sequence(s) provided in ``X``.

            - If ``None``, then ``X`` is assumed to be a single sequence.
            - ``len(X)`` should be equal to ``sum(lengths)``.

        Returns
        -------
        float:
            The Akaike information criterion.

        Notes
        -----
        This method requires a trained model — see :func:`fit`.
        """
        max_log_likelihood = self.model.score(X, lengths=lengths)
        n_params = self.n_params
        return 2 * (n_params - np.log(max_log_likelihood))

    @pyd.validate_call(config=dict(arbitrary_types_allowed=True))
    def set_state_start_probs(
        self: pyd.SkipValidation,
        probs: (
            FloatArray | enums.TransitionMode
        ) = enums.TransitionMode.RANDOM,  # placeholder
        /,
    ) -> None:
        """Set the initial state probabilities.

        If this method is **not** called, initial state probabilities are
        initialized depending on the value of ``topology`` provided to
        :func:`__init__`.

        - If ``topology`` was set to ``'ergodic'``, ``'left-right'`` or
          ``'linear'``, then random probabilities will be assigned
          according to the topology by calling :func:`set_state_start_probs`
          with ``probs='random'``.
        - If ``topology`` was set to ``None``, then initial state
          probabilities will be initialized by
          `hmmlearn <https://hmmlearn.readthedocs.io/en/latest/>`__.

        Parameters
        ----------
        self: BaseHMM

        probs:
            Probabilities or probability type to assign as initial state
            probabilities.

            - If an ``Array``, should be a vector of starting probabilities
              for each state.
            - If ``'uniform'``, there is an equal probability of starting in
              any state.
            - If ``'random'``, the vector of initial state probabilities is
              sampled from a Dirichlet distribution with unit concentration
              parameters.

        Notes
        -----
        If used, this method should normally be called before :func:`fit`.
        """
        if isinstance(probs, enums.TransitionMode):
            self._startprob = probs
            self._skip_init_params |= set("s")
        else:
            self._startprob = np.array(probs, dtype=np.float64)
            self._skip_init_params |= set("s")

    @pyd.validate_call(config=dict(arbitrary_types_allowed=True))
    def set_state_transition_probs(
        self: pyd.SkipValidation,
        probs: (
            FloatArray | enums.TransitionMode
        ) = enums.TransitionMode.RANDOM,  # placeholder
        /,
    ) -> None:
        """Set the transition probability matrix.

        If this method is **not** called, transition probabilities are
        initialized depending on the value of ``topology`` provided to
        :func:`__init__`:

        - If ``topology`` was set to ``'ergodic'``, ``'left-right'`` or
          ``'linear'``, then random probabilities will be assigned according
          to the topology by calling :func:`set_state_transition_probs` with
          ``value='random'``.
        - If ``topology`` was set to ``None``, then initial state
          probabilities will be initialized by
          `hmmlearn <https://hmmlearn.readthedocs.io/en/latest/>`__.

        Parameters
        ----------
        self: BaseHMM

        probs:
            Probabilities or probability type to assign as state transition
            probabilities.

            - If an ``Array``, should be a matrix of probabilities where each
              row must some to one and represents the probabilities of
              transitioning out of a state.
            - If ``'uniform'``, for each state there is an equal probability
              of transitioning to any state permitted by the topology.
            - If ``'random'``, the vector of transition probabilities for
              each row is sampled from a Dirichlet distribution with unit
              concentration parameters, according to the shape of the
              topology.

        Notes
        -----
        If used, this method should normally be called before :func:`fit`.
        """
        if isinstance(probs, enums.TransitionMode):
            self._transmat = probs
            self._skip_init_params |= set("t")
        else:
            self._transmat = np.array(probs, dtype=np.float64)
            self._skip_init_params |= set("t")

    @abc.abstractmethod
    def freeze(self: BaseHMM, params: str | None, /) -> None:
        """Freeze the trainable parameters of the HMM,
        preventing them from be updated during the Baum—Welch algorithm.
        """
        defaults = self._hmmlearn_kwargs_defaults()["params"]
        self._skip_params |= set(self._modify_params(params or defaults))

    @abc.abstractmethod
    def unfreeze(self: BaseHMM, params: str | None, /) -> None:
        """Unfreeze the trainable parameters of the HMM,
        allowing them to be updated during the Baum—Welch algorithm.
        """
        defaults = self._hmmlearn_kwargs_defaults()["params"]
        self._skip_params -= set(self._modify_params(params or defaults))

    def _modify_params(self: BaseHMM, params: str) -> str:
        """Validate parameters to be frozen/unfrozen."""
        defaults = self._hmmlearn_kwargs_defaults()["params"]
        msg = (
            "Expected a string consisting of any combination of "
            f"{defaults!r}"  #
        )
        if isinstance(params, str):
            if bool(re.compile(rf"[^{defaults}]").search(params)):
                raise ValueError(msg)
        else:
            raise TypeError(msg)
        return params

    def _check_init_params(self: BaseHMM) -> None:
        """Validate hmmlearn init_params argument."""
        topology = self.topology_ or _hmm.topologies.ErgodicTopology(
            n_states=self.n_states,
            random_state=check_random_state(self.random_state),
        )

        if "s" in self._skip_init_params:
            if isinstance(self._startprob, enums.TransitionMode):
                if self._startprob == enums.TransitionMode.UNIFORM:
                    self._startprob = topology.uniform_start_probs()
                elif self._startprob == enums.TransitionMode.RANDOM:
                    self._startprob = topology.random_start_probs()
            elif isinstance(self._startprob, np.ndarray):
                self._startprob = topology.check_start_probs(
                    self._startprob,
                )
        elif self.topology_ is not None:
            self.set_state_start_probs(topology.random_start_probs())

        if "t" in self._skip_init_params:
            if isinstance(self._transmat, enums.TransitionMode):
                if self._transmat == enums.TransitionMode.UNIFORM:
                    self._transmat = topology.uniform_transition_probs()
                elif self._transmat == enums.TransitionMode.RANDOM:
                    self._transmat = topology.random_transition_probs()
            elif isinstance(self._transmat, np.ndarray):
                self._transmat = topology.check_transition_probs(
                    self._transmat,
                )
        elif self.topology_ is not None:
            self.set_state_transition_probs(
                topology.random_transition_probs(),
            )

    @classmethod
    def _check_hmmlearn_kwargs(
        cls: type[BaseHMM], kwargs: dict[str, t.Any] | None
    ) -> dict[str, t.Any]:
        """Check hmmlearn forwarded key-word arguments."""
        defaults: dict[str, t.Any] = cls._hmmlearn_kwargs_defaults()
        kwargs: dict[str, t.Any] = kwargs or defaults
        kwargs = copy.deepcopy(kwargs)

        setter_methods = [
            f"{func}()" for func in dir(cls) if func.startswith("set_state")
        ]

        for param in kwargs:
            if param in cls._unsettable_hmmlearn_kwargs():
                if param == "init_params":
                    init_params_defaults = defaults["init_params"]
                    if set(kwargs[param]) != set(init_params_defaults):
                        kwargs[param] = init_params_defaults
                        msg = (
                            "The `init_params` hmmlearn argument cannot be "
                            "overridden manually - defaulting to all "
                            f"parameters {init_params_defaults!r}. "
                            "Use the following method to initialize model "
                            f"parameters: {', '.join(setter_methods)}."
                        )
                        warnings.warn(msg, stacklevel=1)
                elif param == "params":
                    params_defaults = defaults["params"]
                    if set(kwargs[param]) != set(params_defaults):
                        kwargs[param] = params_defaults
                        msg = (
                            "The `params` hmmlearn argument cannot be "
                            "overridden manually - defaulting to all "
                            f"parameters {params_defaults!r}. "
                            "Use the freeze() and unfreeze() methods to "
                            "specify the learnable model parameters."
                        )
                        warnings.warn(msg, stacklevel=1)
                else:
                    del kwargs[param]
                    msg = (
                        f"The {param!r} hmmlearn argument cannot be "
                        f"overridden manually - use the {cls.__name__!r} "
                        "constructor to specify this argument."
                    )
                    warnings.warn(msg, stacklevel=1)

        if "init_params" not in kwargs:
            kwargs["init_params"] = defaults

        if "params" not in kwargs:
            kwargs["params"] = defaults

        return kwargs

    @abc.abstractmethod
    def _init_hmm(self: BaseHMM, **kwargs: t.Any) -> hmmlearn.base.BaseHMM:
        """Initialize the hmmlearn model."""
        raise NotImplementedError

    @abc.abstractstaticmethod
    def _hmmlearn_kwargs_defaults() -> dict[str, t.Any]:
        """Default values for hmmlearn key-word arguments."""
        raise NotImplementedError

    @staticmethod
    def _unsettable_hmmlearn_kwargs() -> list[str]:
        """Arguments that should not be provided in `hmmlearn_kwargs` in
        :func:`__init__`.
        """
        return ["random_state", "init_params", "params"]

    @staticmethod
    def _hmmlearn_params() -> list[str]:
        """Names of trainable hmmlearn parameters."""
        return ["startprob", "transmat"]
