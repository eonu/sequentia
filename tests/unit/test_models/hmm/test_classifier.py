# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

from __future__ import annotations

import copy
import enum
import os
import tempfile
import typing as t
from unittest.mock import Mock

import numpy as np
import pytest
from _pytest.fixtures import SubRequest

from sequentia import enums
from sequentia._internal import _validation
from sequentia.datasets import (
    SequentialDataset,
    load_digits,
    load_gene_families,
)
from sequentia.models.hmm import (
    CategoricalHMM,
    GaussianMixtureHMM,
    HMMClassifier,
)
from sequentia.models.hmm.variants.base import BaseHMM

from .variants.test_categorical import assert_fit as assert_categorical_fit
from .variants.test_gaussian_mixture import (
    assert_fit as assert_gaussian_mixture_fit,
)

n_classes = 7


class FitMode(enum.StrEnum):
    PREFIT = "prefit"
    POSTFIT_IDENTICAL = "postfit_identical"
    POSTFIT_FLEXIBLE = "postfit_flexible"


@pytest.fixture(scope="module")
def random_state(request: SubRequest) -> np.random.RandomState:
    return np.random.RandomState(1)


@pytest.fixture(scope="module")
def dataset(request: SubRequest) -> SequentialDataset | None:
    if request.param == "digits":
        return load_digits(digits=range(n_classes))
    if request.param == "gene_families":
        data, _ = load_gene_families()
        return data
    return None


@pytest.fixture(scope="module")
def model(
    random_state: np.random.RandomState, request: SubRequest
) -> BaseHMM | None:
    if request.param == "gaussian_mixture":
        return GaussianMixtureHMM(
            topology=enums.TopologyMode.LEFT_RIGHT,
            n_states=2,
            n_components=1,
            random_state=random_state,
        )
    if request.param == "categorical":
        return CategoricalHMM(
            topology=enums.TopologyMode.LEFT_RIGHT,
            n_states=2,
            random_state=random_state,
        )
    return None


class MockData:
    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length

    @property
    def lengths(self):
        return MockData(self.length)


def assert_fit(clf: BaseHMM):
    assert hasattr(clf, "prior_")
    assert hasattr(clf, "classes_")
    assert _validation.check_is_fitted(clf, return_=True)

    for hmm in clf.models.values():
        data = MockData(hmm.n_seqs_)
        if isinstance(hmm, GaussianMixtureHMM):
            assert_gaussian_mixture_fit(hmm, data=data)
        elif isinstance(hmm, CategoricalHMM):
            assert_categorical_fit(hmm, data=data)


@pytest.mark.parametrize(
    "model, dataset",  # noqa: PT006
    [("gaussian_mixture", "digits"), ("categorical", "gene_families")],
    indirect=True,
)
@pytest.mark.parametrize(
    "prior",
    [
        enums.PriorMode.UNIFORM,
        enums.PriorMode.FREQUENCY,
        {
            i: (i + 1) / (n_classes * (n_classes + 1) / 2)
            for i in range(n_classes)
        },
    ],
)
@pytest.mark.parametrize("fit_mode", list(FitMode))
@pytest.mark.parametrize("n_jobs", [1, -1])
def test_classifier_e2e(
    request: SubRequest,
    helpers: t.Any,
    model: BaseHMM,
    dataset: SequentialDataset,
    prior: enums.PriorMode | dict[int, float],
    fit_mode: FitMode,
    n_jobs: int,
    random_state: np.random.RandomState,
) -> None:
    clf = HMMClassifier(prior=prior, n_jobs=n_jobs)
    clf.add_models({i: copy.deepcopy(model) for i in range(n_classes)})

    assert clf.prior == prior
    assert len(clf.models) == n_classes
    assert set(clf.models) == set(range(n_classes))
    assert all(isinstance(hmm, type(model)) for hmm in clf.models.values())

    subset, _ = dataset.split(
        test_size=0.6, random_state=random_state, stratify=True
    )
    train, test = subset.split(
        test_size=0.2, random_state=random_state, stratify=True
    )

    if fit_mode == FitMode.PREFIT:
        for X, lengths, c in train.iter_by_class():
            clf.models[c].fit(X, lengths=lengths)
        assert_fit(clf.fit())
    elif fit_mode == FitMode.POSTFIT_FLEXIBLE:
        assert_fit(clf.fit(**train.X_y_lengths))
    elif fit_mode == FitMode.POSTFIT_IDENTICAL:
        clf = HMMClassifier(
            variant=type(model),
            model_kwargs=model.get_params(),
            prior=prior,
            n_jobs=n_jobs,
        )
        clf.fit(**train.X_y_lengths)

    scores_pred = clf.predict_scores(**test.X_lengths)
    assert scores_pred.shape == (len(test), n_classes)

    proba_pred = clf.predict_proba(**test.X_lengths)
    assert proba_pred.shape == (len(test), n_classes)
    helpers.assert_equal(proba_pred.sum(axis=1), 1)
    assert ((proba_pred >= 0) & (proba_pred <= 1)).all()

    y_pred = clf.predict(**test.X_lengths)
    assert y_pred.shape == (len(test),)
    assert set(y_pred).issubset(set(range(n_classes)))

    acc = clf.score(**test.X_y_lengths)
    assert 0 <= acc <= 1

    # check serialization/deserialization
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = f"{temp_dir}/{request.node.originalname}.model"
        # check that save works
        clf.save(model_path)
        assert os.path.isfile(model_path)
        # check that load works
        clf = HMMClassifier.load(model_path)
        # check that loaded model is fitted
        assert_fit(clf)
        y_pred_load = clf.predict(**test.X_lengths)
        # check that predictions are the same as before serialization
        helpers.assert_equal(y_pred, y_pred_load)


@pytest.mark.parametrize("classes", [[0, 1, 2], [2, 0, 1]])
def test_classifier_compute_log_posterior(
    helpers: t.Any, classes: list[int]
) -> None:
    clf = HMMClassifier()
    clf.classes_ = np.array(classes)
    clf.prior_ = {i: np.exp(i) for i in clf.classes_}
    clf.models = {
        i: Mock(score=Mock(side_effect=lambda _: 0)) for i in clf.classes_
    }
    helpers.assert_equal(clf._compute_log_posterior(None), clf.classes_)


def test_classifier_compute_scores_chunk(helpers: t.Any) -> None:
    clf = HMMClassifier()
    clf.classes_ = np.arange(3)
    clf.prior_ = {i: np.exp(i) for i in clf.classes_}
    clf.models = {i: Mock(score=Mock(side_effect=len)) for i in clf.classes_}
    X = np.expand_dims(np.arange(10), axis=-1)
    idxs = np.array([[0, 0], [1, 2], [3, 5], [6, 9]])  # lengths = 0, 1, 2, 3
    helpers.assert_equal(
        clf._compute_scores_chunk(X, idxs=idxs),
        np.tile(np.expand_dims(clf.classes_, axis=-1), len(idxs)).T
        + np.expand_dims(np.arange(len(idxs)), axis=-1),
    )
