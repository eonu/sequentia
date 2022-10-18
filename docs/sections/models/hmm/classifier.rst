.. _hmm_classifier:


HMM Classifier
==============

Multiple HMMs can be combined to form a multi-class classifier.

The :class:`.HMMClassifier` can be used to classify:

- Univariate/multivariate numerical observation sequences, by using :class:`.GaussianMixtureHMM` models.
- Univariate categorical observation sequences, by using :class:`.MultinomialHMM` models.

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

.. autoclass:: sequentia.models.hmm.classifier.HMMClassifier
   :members:
   :inherited-members:
   :exclude-members: get_params, set_params