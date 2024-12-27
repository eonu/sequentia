# Copyright (c) 2019 Sequentia Developers.
# Distributed under the terms of the MIT License (see the LICENSE file).
# SPDX-License-Identifier: MIT
# This source code is part of the Sequentia project (https://github.com/eonu/sequentia).

"""This file is an adapted version of the same file from the
sklearn.model_selection sub-package.

Below is the original license from Scikit-Learn, copied on 27th December 2024
from https://github.com/scikit-learn/scikit-learn/blob/main/COPYING.

---

BSD 3-Clause License

Copyright (c) 2007-2024 The scikit-learn developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>,
#         Gael Varoquaux <gael.varoquaux@normalesup.org>
#         Andreas Mueller <amueller@ais.uni-bonn.de>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Raghav RV <rvraghav93@gmail.com>
# License: BSD 3 clause

import time
import typing as t
from collections import defaultdict
from itertools import product

from sklearn.base import _fit_context, clone, is_classifier
from sklearn.metrics._scorer import _MultimetricScorer
from sklearn.model_selection import _search
from sklearn.model_selection._split import check_cv
from sklearn.model_selection._validation import (
    _insert_error_scores,
    _warn_or_raise_about_fit_failures,
)
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import _check_method_params

from sequentia.model_selection._validation import _fit_and_score

__all__ = ["BaseSearchCV", "GridSearchCV", "RandomizedSearchCV", "param_grid"]


def param_grid(**kwargs: list[t.Any]) -> list[dict[str, t.Any]]:
    """Generates a hyper-parameter grid for a nested object.

    Examples
    --------
    Using :func:`.param_grid` in a grid search to cross-validate over
    settings for :class:`.GaussianMixtureHMM`, which is a nested model
    specified in the constructor of a :class:`.HMMClassifier`. ::

        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import minmax_scale

        from sequentia.enums import PriorMode, CovarianceMode, TopologyMode
        from sequentia.models import HMMClassifier, GaussianMixtureHMM
        from sequentia.preprocessing import IndependentFunctionTransformer
        from sequentia.model_selection import GridSearchCV, StratifiedKFold

        GridSearchCV(
            estimator=Pipeline(
                [
                    ("scale", IndependentFunctionTransformer(minmax_scale)),
                    ("clf", HMMClassifier(variant=GaussianMixtureHMM)),
                ]
            ),
            param_grid={
                "clf__prior": [PriorMode.UNIFORM, PriorMode.FREQUENCY],
                "clf__model_kwargs": param_grid(
                    n_states=[3, 5, 7],
                    n_components=[2, 3, 4],
                    covariance=[
                        CovarianceMode.DIAGONAL, CovarianceMode.SPHERICAL
                    ],
                    topology=[
                        TopologyMode.LEFT_RIGHT, TopologyMode.LINEAR
                    ],
                )
            },
            cv=StratifiedKFold(),
        )

    Parameters
    ----------
    **kwargs:
        Hyper-parameter name and corresponding values.

    Returns
    -------
    Hyper-parameter grid for a nested object.
    """
    return [
        dict(zip(kwargs.keys(), values))
        for values in product(*kwargs.values())
    ]


class BaseSearchCV(_search.BaseSearchCV):
    @_fit_context(
        # *SearchCV.estimator is not validated yet
        prefer_skip_nested_validation=False
    )
    def fit(self, X, y=None, **params):
        """Run fit with all sets of parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_samples, n_samples)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features. For precomputed kernel or
            distance matrix, the expected shape of X is (n_samples, n_samples).

        y : array-like of shape (n_samples, n_output) \
            or (n_samples,), default=None
            Target relative to X for classification or regression;
            None for unsupervised learning.

        **params : dict of str -> object
            Parameters passed to the ``fit`` method of the estimator, the scorer,
            and the CV splitter.

            If a fit parameter is an array-like whose length is equal to
            `num_samples` then it will be split across CV groups along with `X`
            and `y`. For example, the :term:`sample_weight` parameter is split
            because `len(sample_weights) = len(X)`.

        Returns
        -------
        self : object
            Instance of fitted estimator.
        """
        estimator = self.estimator
        scorers, refit_metric = self._get_scorers()

        # X, y = indexable(X, y)  # NOTE @eonu: removed
        params = _check_method_params(X, params=params)

        routed_params = self._get_routed_params_for_fit(params)

        cv_orig = check_cv(self.cv, y, classifier=is_classifier(estimator))
        n_splits = cv_orig.get_n_splits(X, y, **routed_params.splitter.split)

        base_estimator = clone(self.estimator)

        parallel = Parallel(n_jobs=self.n_jobs, pre_dispatch=self.pre_dispatch)

        fit_and_score_kwargs = dict(
            scorer=scorers,
            fit_params=routed_params.estimator.fit,
            score_params=routed_params.scorer.score,
            return_train_score=self.return_train_score,
            return_n_test_samples=True,
            return_times=True,
            return_parameters=False,
            error_score=self.error_score,
            verbose=self.verbose,
        )
        results = {}
        with parallel:
            all_candidate_params = []
            all_out = []
            all_more_results = defaultdict(list)

            def evaluate_candidates(
                candidate_params, cv=None, more_results=None
            ):
                cv = cv or cv_orig
                candidate_params = list(candidate_params)
                n_candidates = len(candidate_params)

                if self.verbose > 0:
                    print(
                        "Fitting {0} folds for each of {1} candidates,"
                        " totalling {2} fits".format(
                            n_splits, n_candidates, n_candidates * n_splits
                        )
                    )

                out = parallel(
                    delayed(_fit_and_score)(
                        clone(base_estimator),
                        X,
                        y,
                        train=train,
                        test=test,
                        parameters=parameters,
                        split_progress=(split_idx, n_splits),
                        candidate_progress=(cand_idx, n_candidates),
                        **fit_and_score_kwargs,
                    )
                    for (cand_idx, parameters), (
                        split_idx,
                        (train, test),
                    ) in product(
                        enumerate(candidate_params),
                        enumerate(
                            cv.split(X, y, **routed_params.splitter.split)
                        ),
                    )
                )

                if len(out) < 1:
                    raise ValueError(
                        "No fits were performed. "
                        "Was the CV iterator empty? "
                        "Were there no candidates?"
                    )
                elif len(out) != n_candidates * n_splits:
                    raise ValueError(
                        "cv.split and cv.get_n_splits returned "
                        f"inconsistent results. Expected {n_splits} "
                        f"splits, got {len(out) // n_candidates}"
                    )

                _warn_or_raise_about_fit_failures(out, self.error_score)

                # For callable self.scoring, the return type is only know after
                # calling. If the return type is a dictionary, the error scores
                # can now be inserted with the correct key. The type checking
                # of out will be done in `_insert_error_scores`.
                if callable(self.scoring):
                    _insert_error_scores(out, self.error_score)

                all_candidate_params.extend(candidate_params)
                all_out.extend(out)

                if more_results is not None:
                    for key, value in more_results.items():
                        all_more_results[key].extend(value)

                nonlocal results
                results = self._format_results(
                    all_candidate_params, n_splits, all_out, all_more_results
                )

                return results

            self._run_search(evaluate_candidates)

            # multimetric is determined here because in the case of a callable
            # self.scoring the return type is only known after calling
            first_test_score = all_out[0]["test_scores"]
            self.multimetric_ = isinstance(first_test_score, dict)

            # check refit_metric now for a callabe scorer that is multimetric
            if callable(self.scoring) and self.multimetric_:
                self._check_refit_for_multimetric(first_test_score)
                refit_metric = self.refit

        # For multi-metric evaluation, store the best_index_, best_params_ and
        # best_score_ iff refit is one of the scorer names
        # In single metric evaluation, refit_metric is "score"
        if self.refit or not self.multimetric_:
            self.best_index_ = self._select_best_index(
                self.refit, refit_metric, results
            )
            if not callable(self.refit):
                # With a non-custom callable, we can select the best score
                # based on the best index
                self.best_score_ = results[f"mean_test_{refit_metric}"][
                    self.best_index_
                ]
            self.best_params_ = results["params"][self.best_index_]

        if self.refit:
            # here we clone the estimator as well as the parameters, since
            # sometimes the parameters themselves might be estimators, e.g.
            # when we search over different estimators in a pipeline.
            # ref: https://github.com/scikit-learn/scikit-learn/pull/26786
            self.best_estimator_ = clone(base_estimator).set_params(
                **clone(self.best_params_, safe=False)
            )

            refit_start_time = time.time()
            if y is not None:
                self.best_estimator_.fit(X, y, **routed_params.estimator.fit)
            else:
                self.best_estimator_.fit(X, **routed_params.estimator.fit)
            refit_end_time = time.time()
            self.refit_time_ = refit_end_time - refit_start_time

            if hasattr(self.best_estimator_, "feature_names_in_"):
                self.feature_names_in_ = self.best_estimator_.feature_names_in_

        # Store the only scorer not as a dict for single metric evaluation
        if isinstance(scorers, _MultimetricScorer):
            self.scorer_ = scorers._scorers
        else:
            self.scorer_ = scorers

        self.cv_results_ = results
        self.n_splits_ = n_splits

        return self


class GridSearchCV(_search.GridSearchCV, BaseSearchCV):
    """Exhaustive search over specified parameter values for an estimator.

    ``cv`` must be a valid splitting method from
    :mod:`sequentia.model_selection`.

    See Also
    --------
    :class:`sklearn.model_selection.GridSearchCV`
        :class:`.GridSearchCV` is a modified version
        of this class that supports sequences.
    """


class RandomizedSearchCV(_search.RandomizedSearchCV, BaseSearchCV):
    """Randomized search on hyper parameters.

    ``cv`` must be a valid splitting method from
    :mod:`sequentia.model_selection`.

    See Also
    --------
    :class:`sklearn.model_selection.RandomizedSearchCV`
        :class:`.RandomizedSearchCV` is a modified version
        of this class that supports sequences.
    """
