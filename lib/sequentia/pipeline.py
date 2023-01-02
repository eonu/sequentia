"""
Pipeline is an adapted version of Pipeline from the sklearn.pipeline module,
and largely relies on its source code.

Below is the original license from Scikit-Learn, copied on 31st December 2022
from https://github.com/scikit-learn/scikit-learn/blob/main/COPYING.

---

BSD 3-Clause License

Copyright (c) 2007-2022 The scikit-learn developers.
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

from __future__ import annotations

from typing import Optional, Any, Union, List, Tuple

import sklearn.pipeline
import sklearn.base
from joblib import Memory

from sklearn.pipeline import _final_estimator_has
from sklearn.utils.metaestimators import available_if
from sklearn.utils import _print_elapsed_time
from sklearn.utils.validation import check_memory
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone

from sequentia.preprocessing.base import Transform
from sequentia.utils.validation import Array

__all__ = ["Pipeline"]


class Pipeline(sklearn.pipeline.Pipeline):
    """Pipeline of transforms with a final estimator.

    Sequentially apply a list of transforms and a final estimator.
    Intermediate steps of the pipeline must be 'transforms', that is, they
    must implement ``fit`` and ``transform`` methods.
    The final estimator only needs to implement ``fit``.
    The transformers in the pipeline can be cached using ``memory`` argument.
    The purpose of the pipeline is to assemble several steps that can be
    cross-validated together while setting different parameters. For this, it
    enables setting parameters of the various steps using their names and the
    parameter name separated by a ``__``. A step's estimator may be replaced
    entirely by setting the parameter with its name to another estimator,
    or a transformer removed by setting it to ``'passthrough'`` or ``None``.

    See Also
    --------
    :class:`sklearn.pipeline.Pipeline`:
        :class:`.Pipeline` is based on :class:`sklearn.pipeline.Pipeline`,
        but adapted to accept and work with sequences.

        Read more in the :ref:`User Guide <pipeline>`.

    Examples
    --------
    Creating a :class:`.Pipeline` consisting of two transforms and a :class:`.KNNClassifier`,
    and fitting it to sequences in the spoken digits dataset. ::

        from sequentia.models import KNNClassifier
        from sequentia.preprocessing import IndependentFunctionTransformer
        from sequentia.pipeline import Pipeline
        from sequentia.datasets import load_digits

        from sklearn.preprocessing import scale
        from sklearn.decomposition import PCA

        # Fetch MFCCs of spoken digits
        digits = load_digits()
        train, test = digits.split(test_size=0.2)

        # Create a pipeline with two transforms and a classifier
        pipeline = Pipeline([
            ('standardize', IndependentFunctionTransformer(scale)),
            ('pca', PCA(n_components=5)),
            ('clf', KNNClassifier(k=1))
        ])

        # Fit the pipeline transforms and classifier to training data
        pipeline.fit(train.X, train.lengths)

        # Apply the transforms to training sequences and make predictions
        y_train_pred = pipeline.predict(train.X, train.y, train.lengths)

        # Calculate accuracy on test data
        acc = pipeline.score(test.X, test.y, test.lengths)
    """

    def __init__(
        self,
        steps: List[Tuple[str, BaseEstimator]],
        *,
        memory: Optional[Union[str, Memory]] = None,
        verbose: bool = False
    ) -> Pipeline:
        """Initializes the :class:`.Pipeline`.

        :param steps: Collection of transforms implementing ``fit``/``transform`` that are chained,
            with the last object being an estimator.

        :param memory: Used to cache the fitted transformers of the pipeline. By default,
            no caching is performed. If a string is given, it is the path to
            the caching directory. Enabling caching triggers a clone of
            the transformers before fitting. Therefore, the transformer
            instance given to the pipeline cannot be inspected
            directly. Use the attribute ``named_steps`` or ``steps`` to
            inspect estimators within the pipeline. Caching the
            transformers is advantageous when fitting is time consuming.

        :param verbose: If ``True``, the time elapsed while fitting each step will be printed as it
            is completed.
        """
        super().__init__(steps, memory=memory, verbose=verbose)


    def _can_transform(self):
        return self._final_estimator == "passthrough" or hasattr(
            self._final_estimator, "transform"
        )


    def _can_inverse_transform(self):
        return all(hasattr(t, "inverse_transform") for _, _, t in self._iter())


    def _fit(
        self,
        X: Array,
        lengths: Optional[Array] = None,
        **fit_params_steps
    ) -> Array:
        # shallow copy of steps - this should really be steps_
        self.steps = list(self.steps)
        self._validate_steps()
        # Setup the memory
        memory = check_memory(self.memory)

        fit_transform_one_cached = memory.cache(_fit_transform_one)

        for step_idx, name, transformer in self._iter(
            with_final=False, filter_passthrough=False
        ):
            if transformer is None or transformer == "passthrough":
                with _print_elapsed_time("Pipeline", self._log_message(step_idx)):
                    continue

            if hasattr(memory, "location") and memory.location is None:
                # we do not clone when caching is disabled to
                # preserve backward compatibility
                cloned_transformer = transformer
            else:
                cloned_transformer = clone(transformer)
            # Fit or load from cache the current transformer
            X, fitted_transformer = fit_transform_one_cached(
                cloned_transformer,
                X,
                lengths,
                None,
                message_clsname="Pipeline",
                message=self._log_message(step_idx),
                **fit_params_steps[name],
            )
            # Replace the transformer of the step with the fitted
            # transformer. This is necessary when loading the transformer
            # from the cache.
            self.steps[step_idx] = (name, fitted_transformer)
        return X


    def fit(
        self,
        X: Array,
        y: Optional[Array] = None,
        lengths: Optional[Array] = None,
        **fit_params
    ) -> Pipeline:
        """Fit the model.

        Fit all the transformers one after the other and transform the
        data. Finally, fit the transformed data using the final estimator.

        :param X: Univariate or multivariate observation sequence(s).

            - Should be a single 1D or 2D array.
            - Should have length as the 1st dimension and features as the 2nd dimension.
            - Should be a concatenated sequence if multiple sequences are provided,
              with respective sequence lengths being provided in the ``lengths`` argument for decoding the original sequences.

        :param y: Outputs corresponding to sequence(s) provided in ``X``.
            Only required if the final estimator is a supervised model.

        :param lengths: Lengths of the observation sequence(s) provided in ``X``.

            - If ``None``, then ``X`` is assumed to be a single observation sequence.
            - ``len(X)`` should be equal to ``sum(lengths)``.

        :param fit_params: Parameters passed to the ``fit`` method of each step,
            where each parameter name is prefixed such that parameter ``p`` for step ``s`` has key ``s__p``.

        :return: The fitted pipeline.
        """
        fit_params_steps = self._check_fit_params(**fit_params)
        Xt = self._fit(X, lengths, **fit_params_steps)

        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if self._final_estimator != "passthrough":
                fit_params_last_step = fit_params_steps[self.steps[-1][0]]
                if isinstance(self._final_estimator, ClassifierMixin):
                    self._final_estimator.fit(Xt, y, lengths, **fit_params_last_step)
                else:
                    self._final_estimator.fit(Xt, lengths, **fit_params_last_step)
        return self


    def fit_transform(
        self,
        X: Array,
        lengths: Optional[Array] = None,
        **fit_params
    ) -> Array:
        """Fit the model and transform with the final estimator.

        Fits all the transformers one after the other and transform the
        data. Then uses `fit_transform` on transformed data with the final
        estimator.

        :param X: Univariate or multivariate observation sequence(s).

            - Should be a single 1D or 2D array.
            - Should have length as the 1st dimension and features as the 2nd dimension.
            - Should be a concatenated sequence if multiple sequences are provided,
              with respective sequence lengths being provided in the ``lengths`` argument for decoding the original sequences.

        :param lengths: Lengths of the observation sequence(s) provided in ``X``.

            - If ``None``, then ``X`` is assumed to be a single observation sequence.
            - ``len(X)`` should be equal to ``sum(lengths)``.

        :param fit_params: Parameters passed to the ``fit`` method of each step,
            where each parameter name is prefixed such that parameter ``p`` for step ``s`` has key ``s__p``.

        :return: The transformed data.
        """
        fit_params_steps = self._check_fit_params(**fit_params)
        Xt = self._fit(X, lengths, **fit_params_steps)

        last_step = self._final_estimator
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if last_step == "passthrough":
                return Xt
            fit_params_last_step = fit_params_steps[self.steps[-1][0]]

            if is_sequentia_transform := isinstance(last_step, Transform):
                fit_params_last_step["lengths"] = lengths

            if hasattr(last_step, "fit_transform"):
                return last_step.fit_transform(Xt, **fit_params_last_step)
            else:
                transform_params = {}
                if is_sequentia_transform:
                    transform_params["lengths"] = lengths
                getattr(last_step, "transform")
                return last_step.fit(Xt, **fit_params_last_step).transform(Xt, **transform_params)


    @available_if(_final_estimator_has("predict"))
    def predict(
        self,
        X: Array,
        lengths: Optional[Array] = None
    ) -> Array:
        """Transform the data, and apply `predict` with the final estimator.

        Call `transform` of each transformer in the pipeline. The transformed
        data are finally passed to the final estimator that calls `predict`
        method. Only valid if the final estimator implements `predict`.

        :param X: Univariate or multivariate observation sequence(s).

            - Should be a single 1D or 2D array.
            - Should have length as the 1st dimension and features as the 2nd dimension.
            - Should be a concatenated sequence if multiple sequences are provided,
              with respective sequence lengths being provided in the ``lengths`` argument for decoding the original sequences.

        :param lengths: Lengths of the observation sequence(s) provided in ``X``.

            - If ``None``, then ``X`` is assumed to be a single observation sequence.
            - ``len(X)`` should be equal to ``sum(lengths)``.

        :return: Output predictions.
        """
        Xt = X
        for _, name, transform in self._iter(with_final=False):
            transform_params = {}
            if isinstance(transform, Transform):
                transform_params["lengths"] = lengths
            Xt = transform.transform(Xt, **transform_params)
        return self.steps[-1][1].predict(Xt, lengths)


    @available_if(_final_estimator_has("fit_predict"))
    def fit_predict(
        self,
        X: Array,
        y: Array,
        lengths: Optional[Array] = None,
        **fit_params
    ) -> Array:
        """Transform the data, and apply `fit_predict` with the final estimator.

        Call `fit_transform` of each transformer in the pipeline. The
        transformed data are finally passed to the final estimator that calls
        `fit_predict` method. Only valid if the final estimator implements
        `fit_predict`.

        :param X: Univariate or multivariate observation sequence(s).

            - Should be a single 1D or 2D array.
            - Should have length as the 1st dimension and features as the 2nd dimension.
            - Should be a concatenated sequence if multiple sequences are provided,
              with respective sequence lengths being provided in the ``lengths`` argument for decoding the original sequences.

        :param y: Outputs corresponding to sequence(s) provided in ``X``.

        :param lengths: Lengths of the observation sequence(s) provided in ``X``.

            - If ``None``, then ``X`` is assumed to be a single observation sequence.
            - ``len(X)`` should be equal to ``sum(lengths)``.

        :param fit_params: Parameters passed to the ``fit`` method of each step,
            where each parameter name is prefixed such that parameter ``p`` for step ``s`` has key ``s__p``.

        :return: Output predictions.
        """
        fit_params_steps = self._check_fit_params(**fit_params)
        Xt = self._fit(X, lengths, **fit_params_steps)

        fit_params_last_step = fit_params_steps[self.steps[-1][0]]
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            y_pred = self.steps[-1][1].fit_predict(Xt, y, lengths, **fit_params_last_step)
        return y_pred


    @available_if(_final_estimator_has("predict_proba"))
    def predict_proba(
        self,
        X: Array,
        lengths: Optional[Array] = None
    ) -> Array:
        """Transform the data, and apply `predict_proba` with the final estimator.

        Call `transform` of each transformer in the pipeline. The transformed
        data are finally passed to the final estimator that calls
        `predict_proba` method. Only valid if the final estimator implements
        `predict_proba`.

        :param X: Univariate or multivariate observation sequence(s).

            - Should be a single 1D or 2D array.
            - Should have length as the 1st dimension and features as the 2nd dimension.
            - Should be a concatenated sequence if multiple sequences are provided,
              with respective sequence lengths being provided in the ``lengths`` argument for decoding the original sequences.

        :param lengths: Lengths of the observation sequence(s) provided in ``X``.

            - If ``None``, then ``X`` is assumed to be a single observation sequence.
            - ``len(X)`` should be equal to ``sum(lengths)``.

        :return: Output probabilities.
        """
        Xt = X
        for _, name, transform in self._iter(with_final=False):
            transform_params = {}
            if isinstance(transform, Transform):
                transform_params["lengths"] = lengths
            Xt = transform.transform(Xt, **transform_params)
        return self.steps[-1][1].predict_proba(Xt, lengths)


    @available_if(_can_transform)
    def transform(
        self,
        X: Array,
        lengths: Optional[Array] = None
    ) -> Array:
        """Transform the data, and apply `transform` with the final estimator.

        Call `transform` of each transformer in the pipeline. The transformed
        data are finally passed to the final estimator that calls
        `transform` method. Only valid if the final estimator
        implements `transform`.

        This also works where final estimator is `None` in which case all prior
        transformations are applied.

        :param X: Univariate or multivariate observation sequence(s).

            - Should be a single 1D or 2D array.
            - Should have length as the 1st dimension and features as the 2nd dimension.
            - Should be a concatenated sequence if multiple sequences are provided,
              with respective sequence lengths being provided in the ``lengths`` argument for decoding the original sequences.

        :param lengths: Lengths of the observation sequence(s) provided in ``X``.

            - If ``None``, then ``X`` is assumed to be a single observation sequence.
            - ``len(X)`` should be equal to ``sum(lengths)``.

        :return: The transformed data.
        """
        Xt = X
        for _, _, transform in self._iter():
            transform_params = {}
            if isinstance(transform, Transform):
                transform_params["lengths"] = lengths
            Xt = transform.transform(Xt, **transform_params)
        return Xt


    @available_if(_can_inverse_transform)
    def inverse_transform(
        self,
        X: Array,
        lengths: Optional[Array] = None
    ) -> Array:
        """Apply `inverse_transform` for each step in a reverse order.

        All estimators in the pipeline must support `inverse_transform`.

        :param X: Univariate or multivariate observation sequence(s).

            - Should be a single 1D or 2D array.
            - Should have length as the 1st dimension and features as the 2nd dimension.
            - Should be a concatenated sequence if multiple sequences are provided,
              with respective sequence lengths being provided in the ``lengths`` argument for decoding the original sequences.

        :param lengths: Lengths of the observation sequence(s) provided in ``X``.

            - If ``None``, then ``X`` is assumed to be a single observation sequence.
            - ``len(X)`` should be equal to ``sum(lengths)``.

        :return: The inverse transformed data.
        """
        reverse_iter = reversed(list(self._iter()))
        for _, _, transform in reverse_iter:
            transform_params = {}
            if isinstance(transform, Transform):
                transform_params["lengths"] = lengths
            X = transform.inverse_transform(X, **transform_params)
        return X


    @available_if(_final_estimator_has("score"))
    def score(
        self,
        X: Array,
        y: Optional[Array] = None,
        lengths: Optional[Array] = None,
        sample_weight: Optional[Any] = None
    ) -> float:
        """Transform the data, and apply `score` with the final estimator.

        Call `transform` of each transformer in the pipeline. The transformed
        data are finally passed to the final estimator that calls
        `score` method. Only valid if the final estimator implements `score`.

        :param X: Univariate or multivariate observation sequence(s).

            - Should be a single 1D or 2D array.
            - Should have length as the 1st dimension and features as the 2nd dimension.
            - Should be a concatenated sequence if multiple sequences are provided,
              with respective sequence lengths being provided in the ``lengths`` argument for decoding the original sequences.

        :param y: Outputs corresponding to sequence(s) provided in ``X``.
            Must be provided if the final estimator is a model, i.e. not a transform.

        :param lengths: Lengths of the observation sequence(s) provided in ``X``.

            - If ``None``, then ``X`` is assumed to be a single observation sequence.
            - ``len(X)`` should be equal to ``sum(lengths)``.

        :param sample_weight: If not ``None``, this argument is passed as ``sample_weight``
            keyword argument to the ``score`` method of the final estimator.

        :return: Result of calling `score` on the final estimator.
        """
        Xt = X
        for _, name, transform in self._iter(with_final=False):
            transform_params = {}
            if isinstance(transform, Transform):
                transform_params["lengths"] = lengths
            Xt = transform.transform(Xt, **transform_params)

        score_params = {}
        if sample_weight is not None:
            score_params["sample_weight"] = sample_weight

        last_step = self.steps[-1][1]
        if isinstance(last_step, TransformerMixin):
            return last_step.score(Xt, lengths, **score_params)
        else:
            return last_step.score(Xt, y, lengths, **score_params)


def _fit_transform_one(
    transformer,
    X,
    lengths,
    weight,
    message_clsname="",
    message=None,
    **fit_params
):
    with _print_elapsed_time(message_clsname, message):
        if is_sequentia_transformer := isinstance(transformer, Transform):
            fit_params["lengths"] = lengths
        if hasattr(transformer, "fit_transform"):
            res = transformer.fit_transform(X, **fit_params)
        else:
            transform_params = {}
            if is_sequentia_transformer:
                transform_params["lengths"] = lengths
            res = transformer.fit(X, **fit_params).transform(X, **transform_params)

    if weight is None:
        return res, transformer
    return res * weight, transformer


if __name__ == "__main__":
    import numpy as np

    from sequentia.models import KNNClassifier, HMMClassifier, GaussianMixtureHMM
    from sequentia.datasets import load_digits
    from sequentia.preprocessing import IndependentFunctionTransformer
    # try normal FunctionTransformer

    from sklearn.preprocessing import StandardScaler, scale
    from sklearn.decomposition import PCA

    random_state = np.random.RandomState(0)

    # digits = load_digits(digits=[0, 1])
    digits = load_digits()
    # subset, _ = digits.split(train_size=0.1, stratify=True, random_state=random_state)
    # train, test = subset.split(test_size=0.2)
    train, test = digits.split(test_size=0.2)

    pipeline = Pipeline([
        ('standardize', IndependentFunctionTransformer(scale)),
        ('pca', PCA(n_components=5, random_state=random_state)),
        ('clf', HMMClassifier(n_jobs=-1).add_models({
            k: GaussianMixtureHMM(random_state=random_state)
            for k in train.classes
        }))
        # ('clf', KNNClassifier(k=1, use_c=True, n_jobs=-1, random_state=random_state))
    ])

    # Xt = pipeline.fit_transform(*train.X_lengths)

    y_pred = pipeline.fit_predict(*train.X_y_lengths)

    breakpoint()

    # from scipy.signal import medfilt2d, convolve
    # ('median_filter', IndependentFunctionTransformer(medfilt2d, kw_args={"kernel_size": (5, 1)}))
