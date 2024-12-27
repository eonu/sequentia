# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""A classifier consisting of HMMs, each trained independently to recognize
sequences of a single class.
"""

from __future__ import annotations

import pathlib
import typing as t

import joblib
import numpy as np
import pydantic as pyd
from sklearn.utils.validation import NotFittedError

from sequentia._internal import _data, _multiprocessing, _sklearn, _validation
from sequentia._internal._typing import Array, FloatArray, IntArray
from sequentia.datasets.base import SequentialDataset
from sequentia.enums import PriorMode
from sequentia.models.base import ClassifierMixin
from sequentia.models.hmm import variants


class HMMClassifier(ClassifierMixin):
    """A classifier consisting of HMMs, each trained independently to
    recognize sequences of a single class.

    The predicted class for a given observation sequence is the class
    represented by the HMM which produces the maximum posterior
    probability for the observation sequence.

    Examples
    --------
    Using a :class:`.HMMClassifier` with :class:`.GaussianMixtureHMM`
    models for each class (all with identical settings),
    to classify spoken digits. ::

        import numpy as np
        from sequentia.datasets import load_digits
        from sequentia.models.hmm import GaussianMixtureHMM, HMMClassifier

        # Seed for reproducible pseudo-randomness
        random_state = np.random.RandomState(1)

        # Fetch MFCCs of spoken digits
        data = load_digits()
        train_data, test_data = data.split(
            test_size=0.2, random_state=random_state
        )

        # Create a HMMClassifier using:
        # - a separate GaussianMixtureHMM for each class (with 3 states)
        # - a class frequency prior
        clf = HMMClassifier(
            variant=GaussianMixtureHMM,
            model_kwargs=dict(n_states=3, random_state=random_state)
            prior='frequency',
        )

        # Fit the HMMs by providing observation sequences for all classes
        clf.fit(train_data.X, train_data.y, lengths=train_data.lengths)

        # Predict classes for the test observation sequences
        y_pred = clf.predict(test_data.X, lengths=test_data.lengths)

    For more complex problems, it might be necessary to specify different
    hyper-parameters for each individual class HMM. This can be done by
    using :func:`add_model` or :func:`add_models` to add HMM objects
    after the :class:`HMMClassifier` has been initialized. ::

        # Create a HMMClassifier using a class frequency prior
        clf = HMMClassifier(prior='frequency')

        # Add an untrained HMM for each class
        for label in data.classes:
            model = GaussianMixtureHMM(random_state=random_state)
            clf.add_model(model, label=label)

        # Fit the HMMs by providing observation sequences for all classes
        clf.fit(train_data.X, train_data.y, lengths=train_data.lengths)

    Alternatively, we might want to pre-fit the HMMs individually,
    then add these fitted HMMs to the :class:`.HMMClassifier`. In this case,
    :func:`fit` on the :class:`.HMMClassifier` is called without providing any
    data as arguments, since the HMMs are already fitted. ::

        # Create a HMMClassifier using a class frequency prior
        clf = HMMClassifier(prior='frequency')

        # Manually fit each HMM on its own subset of data
        for X_train, lengths_train, label for train_data.iter_by_class():
            model = GaussianMixtureHMM(random_state=random_state)
            model.fit(X_train, lengths=lengths_train)
            clf.add_model(model, label=label)

        # Fit the classifier
        clf.fit()
    """

    @pyd.validate_call(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self: pyd.SkipValidation,
        *,
        variant: type[variants.CategoricalHMM]
        | type[variants.GaussianMixtureHMM]
        | None = None,
        model_kwargs: dict[str, t.Any] | None = None,
        prior: (
            PriorMode | dict[int, pyd.confloat(ge=0, le=1)]
        ) = PriorMode.UNIFORM,  # placeholder
        classes: list[int] | None = None,
        n_jobs: pyd.PositiveInt | pyd.NegativeInt = 1,
    ) -> pyd.SkipValidation:
        """Initialize a :class:`.HMMClassifier`.

        Parameters
        ----------
        self: HMMClassifier

        variant:
            Variant of HMM to use for modelling each class. If not specified,
            models must instead be added using the :func:`add_model` or
            :func:`add_models` methods after the :class:`.HMMClassifier` has
            been initialized.

        model_kwargs:
            If ``variant`` is specified, these parameters are used to
            initialize the created HMM object(s). Note that all HMMs
            will be created with identical settings.

        prior:
            Type of prior probability to assign to each HMM.

            - If ``"uniform"``, a uniform prior will be used, making each HMM
              equally likely.
            - If ``"frequency"``, the prior probability of each HMM is equal
              to the fraction of total observation sequences that the HMM was
              fitted with.
            - If a ``dict``, custom prior probabilities can be assigned to
              each HMM. The keys should be the label of the class represented
              by the HMM, and the value should be the prior probability for
              the HMM.

        classes:
            Set of possible class labels.

            - If not provided, these will be determined from the training
              data labels.
            - If provided, output from methods such as :func:`predict_proba`
              and :func:`predict_scores` will follow the ordering of the
              class labels provided here.

        n_jobs:
            Maximum number of concurrently running workers.

            - If 1, no parallelism is used at all (useful for debugging).
            - If -1, all CPUs are used.
            - If < -1, ``(n_cpus + 1 + n_jobs)`` are used — e.g.
              ``n_jobs=-2`` uses all but one.

        Returns
        -------
        HMMClassifier
        """
        #: Type of HMM to use for each class.
        self.variant: (
            type[variants.CategoricalHMM]
            | type[variants.GaussianMixtureHMM]
            | None
        ) = variant
        #: Model parameters for initializing HMMs.
        self.model_kwargs: dict[str, t.Any] | None = model_kwargs
        #: Type of prior probability to assign to each HMM.
        self.prior: PriorMode | dict[int, pyd.confloat(ge=0, le=1)] = prior
        #: Set of possible class labels.
        self.classes: list[int] | None = classes
        #: Maximum number of concurrently running workers.
        self.n_jobs: pyd.PositiveInt | pyd.NegativeInt = n_jobs
        #: HMMs constituting the :class:`.HMMClassifier`.
        self.models: dict[int, variants.BaseHMM] = {}

        # Allow metadata routing for lengths
        if _sklearn.routing_enabled():
            self.set_fit_request(lengths=True)
            self.set_predict_request(lengths=True)
            self.set_predict_proba_request(lengths=True)
            self.set_predict_log_proba_request(lengths=True)
            self.set_score_request(
                lengths=True,
                normalize=True,
                sample_weight=True,
            )

    @pyd.validate_call(config=dict(arbitrary_types_allowed=True))
    def add_model(
        self: pyd.SkipValidation,
        model: variants.BaseHMM,
        /,
        *,
        label: int,
    ) -> pyd.SkipValidation:
        """Add a single HMM to the classifier.

        Parameters
        ----------
        self: HMMClassifier

        model:
            HMM to add to the classifier.

        label:
            Class represented by the HMM.

        Returns
        -------
        HMMClassifier
            The classifier.

        Notes
        -----
        All models added to the classifier must be of the same type — either
        :class:`.GaussianMixtureHMM` or :class:`.CategoricalHMM`.
        """
        if len(self.models) > 0 and not isinstance(
            model, type(next(iter(self.models.values())))
        ):
            msg = (
                f"Model of type {type(model).__name__} must be the same "
                "as the models already provided to this "
                f"{type(self).__name__} instance"
            )
            raise TypeError(msg)
        self.models[int(label)] = model
        return self

    @pyd.validate_call(config=dict(arbitrary_types_allowed=True))
    def add_models(
        self: pyd.SkipValidation,
        models: dict[int, variants.BaseHMM],
        /,
    ) -> pyd.SkipValidation:
        """Add HMMs to the classifier.

        Parameters
        ----------
        self: HMMClassifier

        models:
            HMMs to add to the classifier. The key for each HMM should be the
            label of the class represented by the HMM.

        Returns
        -------
        HMMClassifier
            The classifier.

        Notes
        -----
        All models added to the classifier must be of the same type — either
        :class:`.GaussianMixtureHMM` or :class:`.CategoricalHMM`.
        """
        for label, model in models.items():
            self.add_model(model, label=label)
        return self

    def fit(
        self: HMMClassifier,
        X: Array | None = None,
        y: IntArray | None = None,
        *,
        lengths: IntArray | None = None,
    ) -> HMMClassifier:
        """Fit the HMMs to the sequence(s) in ``X``.

        - If fitted models were provided with :func:`add_model` or
          :func:`add_models`, no arguments should be passed to :func:`fit`.
        - If unfitted models were provided with :func:`add_model` or
          :func:`add_models`, or a ``variant`` was specified in
          :func:`HMMClassifier.__init__`, training data ``X``, ``y`` and
          ``lengths`` must be provided to :func:`fit`.

        Parameters
        ----------
        self: HMMClassifier

        X:
            Sequence(s).

        y:
            Classes corresponding to sequence(s) in ``X``.

        lengths:
            Lengths of the sequence(s) provided in ``X``.

            - If ``None``, then ``X`` is assumed to be a single sequence.
            - ``len(X)`` should be equal to ``sum(lengths)``.

        Returns
        -------
        HMMClassifier
            The fitted classifier
        """
        if X is None or y is None:
            if len(self.models) == 0:
                msg = (
                    "Fitted models must be provided if no training data is "
                    "provided - use add_model() to add fitted models to the "
                    "classifier object"
                )
                raise RuntimeError(msg)

            for label, model in self.models.items():
                if not _validation.check_is_fitted(model, return_=True):
                    msg = (
                        f"The model corresponding to label {label} must be "
                        "pre-fitted if no training data is provided"
                    )
                    raise NotFittedError(msg)

            if self.classes is not None:
                self._classes = _validation.check_classes(
                    self.classes, classes=self.classes
                )
            else:
                # Fetch classes from provided models
                self.classes_ = np.array(list(self.models.keys()))
        else:
            y = _validation.check_y(y, lengths=lengths, dtype=np.int8)
            self.classes_ = _validation.check_classes(y, classes=self.classes)

        # Initialize models based on instructor spec if provided
        if self.variant:
            model_kwargs = self.model_kwargs or {}
            self.models = {
                label: self.variant(**model_kwargs) for label in self.classes_
            }

        # Check that each label has a HMM (and vice versa)
        if set(self.models.keys()) != set(self.classes_):
            msg = (
                "Classes in the dataset are not consistent with the added "
                "models - ensure that every added model corresponds to a "
                "class in the dataset"
            )
            raise ValueError(msg)

        if X is not None and y is not None:
            # Iterate through dataset by class and fit the corresponding model
            dataset = SequentialDataset(
                X,
                y,
                lengths=lengths,
                classes=self.classes_,
            )

            # get number of jobs
            n_jobs = _multiprocessing.effective_n_jobs(
                self.n_jobs, x=self.classes_
            )

            # fit models in parallel
            self.models = dict(
                zip(
                    self.classes_,
                    joblib.Parallel(n_jobs=n_jobs, max_nbytes=None)(
                        joblib.delayed(self.models[c].fit)(
                            X_c, lengths=lengths_c
                        )
                        for X_c, lengths_c, c in dataset.iter_by_class()
                    ),
                )
            )

        # Set class priors
        models: t.Iterable[int, variants.BaseHMM] = self.models.items()
        if self.prior == PriorMode.UNIFORM:
            self.prior_ = {c: 1 / len(self.classes_) for c, _ in models}
        elif self.prior == PriorMode.FREQUENCY:
            total_seqs = sum(mod.n_seqs_ for _, mod in models)
            self.prior_ = {c: mod.n_seqs_ / total_seqs for c, mod in models}
        elif isinstance(self.prior, dict):
            if set(self.prior.keys()) != set(self.classes_):
                msg = (
                    "Classes in the dataset are not consistent with the "
                    "classes in `prior` - ensure that every provided class "
                    "prior corresponds to a class in the dataset"
                )
                raise ValueError(msg)
            self.prior_ = self.prior

        return self

    @_validation.requires_fit
    def predict(
        self: HMMClassifier,
        X: Array,
        *,
        lengths: IntArray | None = None,
    ) -> IntArray:
        """Predict classes for the sequence(s) in ``X``.

        Parameters
        ----------
        self: HMMClassifier

        X:
            Sequence(s).

        lengths:
            Lengths of the sequence(s) provided in ``X``.

            - If ``None``, then ``X`` is assumed to be a single sequence.
            - ``len(X)`` should be equal to ``sum(lengths)``.

        Returns
        -------
        numpy.ndarray
            Class predictions.

        Notes
        -----
        This method requires a trained classifier — see :func:`fit`.
        """
        scores = self.predict_scores(X, lengths=lengths)
        max_score_idxs = scores.argmax(axis=1)
        return self.classes_[max_score_idxs]

    @_validation.requires_fit
    def predict_log_proba(
        self: HMMClassifier, X: Array, *, lengths: IntArray | None = None
    ) -> FloatArray:
        """Predict log un-normalized posterior probabilities for the
        sequences in ``X``.

        Parameters
        ----------
        self: HMMClassifier

        X:
            Sequence(s).

        lengths:
            Lengths of the sequence(s) provided in ``X``.

            - If ``None``, then ``X`` is assumed to be a single sequence.
            - ``len(X)`` should be equal to ``sum(lengths)``.

        Returns
        -------
        numpy.ndarray:
            Log probabilities.

        Notes
        -----
        This method requires a trained classifier — see :func:`fit`.
        """
        return self.predict_scores(X, lengths=lengths)

    @_validation.requires_fit
    def predict_proba(
        self: HMMClassifier, X: Array, *, lengths: IntArray | None = None
    ) -> FloatArray:
        """Predict class probabilities for the sequence(s) in ``X``.

        Probabilities are calculated as the posterior probability of each
        HMM generating the sequence.

        Parameters
        ----------
        self: HMMClassifier

        X:
            Sequence(s).

        lengths:
            Lengths of the sequence(s) provided in ``X``.

            - If ``None``, then ``X`` is assumed to be a single sequence.
            - ``len(X)`` should be equal to ``sum(lengths)``.

        Returns
        -------
        numpy.ndarray:
            Class membership probabilities.

        Notes
        -----
        This method requires a trained classifier — see :func:`fit`.
        """
        proba = self.predict_log_proba(X, lengths=lengths)
        proba -= proba.max(axis=1, keepdims=True)
        proba = np.exp(proba)
        proba /= proba.sum(axis=1, keepdims=True)
        return proba

    @_validation.requires_fit
    def predict_scores(
        self: HMMClassifier, X: Array, *, lengths: IntArray | None = None
    ) -> FloatArray:
        """Predict class scores for the sequence(s) in ``X``.

        Scores are calculated as the log posterior probability of each HMM
        generating the sequence.

        Parameters
        ----------
        self: HMMClassifier

        X:
            Sequence(s).

        lengths:
            Lengths of the sequence(s) provided in ``X``.

            - If ``None``, then ``X`` is assumed to be a single sequence.
            - ``len(X)`` should be equal to ``sum(lengths)``.

        Returns
        -------
        numpy.ndarray:
            Class scores.

        Notes
        -----
        This method requires a trained classifier — see :func:`fit`.
        """
        model: variants.BaseHMM = next(iter(self.models.values()))
        X, lengths = _validation.check_X_lengths(
            X,
            lengths=lengths,
            dtype=model._DTYPE,  # noqa: SLF001
        )
        n_jobs = _multiprocessing.effective_n_jobs(self.n_jobs, x=lengths)
        chunk_idxs = np.array_split(_data.get_idxs(lengths), n_jobs)
        return np.concatenate(
            joblib.Parallel(n_jobs=n_jobs, max_nbytes=None)(
                joblib.delayed(self._compute_scores_chunk)(X, idxs=idxs)
                for idxs in chunk_idxs
            )
        )

    @_validation.requires_fit
    def save(self: HMMClassifier, path: str | pathlib.Path | t.IO, /) -> None:
        """Serialize and save a fitted HMM classifier.

        Parameters
        ----------
        self: HMMClassifier

        path:
            Location to save the serialized classifier.

        Notes
        -----
        This method requires a trained classifier — see :func:`fit`.

        See Also
        --------
        load:
            Load and deserialize a fitted HMM classifier.
        """
        # Fetch main parameters and fitted values
        dict_ = self.__dict__.items()
        state = {
            "params": self.get_params(),
            "models": self.models,
            "fitted": {k: v for k, v in dict_ if k.endswith("_")},
        }

        # Serialize model
        joblib.dump(state, path)

    @classmethod
    def load(
        cls: type[HMMClassifier],
        path: str | pathlib.Path | t.IO,
        /,
    ) -> HMMClassifier:
        """Load and deserialize a fitted HMM classifier.

        Parameters
        ----------
        cls: type[HMMClassifier]

        path:
            Location to load the serialized classifier from.

        Returns
        -------
        HMMClassifier
            Fitted HMM classifier.

        See Also
        --------
        save:
            Serialize and save a fitted HMM classifier.
        """
        state = joblib.load(path)

        # Set main parameters
        model = cls(**state["params"])
        model.models = state["models"]

        # Set fitted values
        for k, v in state["fitted"].items():
            setattr(model, k, v)

        # Return deserialized model
        return model

    def _compute_scores_chunk(
        self: HMMClassifier, X: Array, /, *, idxs: IntArray
    ) -> FloatArray:
        """Compute log posterior probabilities for a chunk of sequences."""
        scores = np.zeros((len(idxs), len(self.classes_)))
        for i, x in enumerate(_data.iter_X(X, idxs=idxs)):
            scores[i] = self._compute_log_posterior(x)
        return scores

    def _compute_log_posterior(
        self: HMMClassifier,
        x: Array,
        /,
    ) -> FloatArray:
        """Compute log posterior probabilities for each class."""
        log_posterior = np.full(len(self.classes_), -np.inf)
        for i, k in enumerate(self.classes_):
            model = self.models[k]
            log_prior = np.log(self.prior_[k])
            log_likelihood = model.score(x)
            log_posterior[i] = log_prior + log_likelihood
        return log_posterior
