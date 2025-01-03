.. _hmm_classifier:


HMM Classifier
==============

Multiple HMMs can be combined to form a multi-class classifier.

The :class:`.HMMClassifier` can be used to classify:

- Univariate/multivariate numerical observation sequences, by using :class:`.GaussianMixtureHMM` models.
- Univariate categorical observation sequences, by using :class:`.CategoricalHMM` models.

To classify a new observation sequence :math:`O'`, the :class:`.HMMClassifier` works by:

1. | Creating HMMs :math:`\lambda_1, \lambda_2, \ldots, \lambda_C` and training each model using the Baum—Welch algorithm on the subset of training observation sequences with the same class label as the model.

2. | Calculating the likelihoods :math:`\mathbb{P}(O'\ |\ \lambda_1), \mathbb{P}(O'\ |\ \lambda_2), \ldots, \mathbb{P}(O'\ |\ \lambda_C)` of each model generating :math:`O'` using the Forward algorithm.

3. | Scaling the likelihoods by priors :math:`p(\lambda_1), p(\lambda_2), \ldots, p(\lambda_C)`, producing un-normalized posteriors
   | :math:`\mathbb{P}(O'\ |\ \lambda_c)p(\lambda_c)` for each class.

4. | Choosing the class represented by the HMM with the highest posterior probability for :math:`O'`.

   .. math::

      c' = \operatorname*{\arg\max}_{c\in\{1,\ldots,C\}}\ p(\lambda_c\ |\ O')
         = \operatorname*{\arg\\max}_{c\in\{1,\ldots,C\}}\ \mathbb{P}(O'\ |\ \lambda_c)p(\lambda_c)

These steps are summarized in the diagram below.

.. image:: /_static/images/classifier.png
    :alt: HMM Classifier
    :width: 80%
    :align: center

API reference
-------------

Class
^^^^^

.. autosummary::

   ~sequentia.models.hmm.classifier.HMMClassifier

Methods
^^^^^^^

.. autosummary::

   ~sequentia.models.hmm.classifier.HMMClassifier.__init__
   ~sequentia.models.hmm.classifier.HMMClassifier.add_model
   ~sequentia.models.hmm.classifier.HMMClassifier.add_models
   ~sequentia.models.hmm.classifier.HMMClassifier.fit
   ~sequentia.models.hmm.classifier.HMMClassifier.fit_predict
   ~sequentia.models.hmm.classifier.HMMClassifier.load
   ~sequentia.models.hmm.classifier.HMMClassifier.predict
   ~sequentia.models.hmm.classifier.HMMClassifier.predict_log_proba
   ~sequentia.models.hmm.classifier.HMMClassifier.predict_proba
   ~sequentia.models.hmm.classifier.HMMClassifier.predict_scores
   ~sequentia.models.hmm.classifier.HMMClassifier.save
   ~sequentia.models.hmm.classifier.HMMClassifier.score

.. _definitions:

Definitions
^^^^^^^^^^^

.. autoclass:: sequentia.models.hmm.classifier.HMMClassifier
   :members:
   :inherited-members:
   :exclude-members: get_params, set_params, get_metadata_routing, set_fit_request, set_predict_log_proba_request, set_predict_proba_request, set_predict_request, set_score_request
