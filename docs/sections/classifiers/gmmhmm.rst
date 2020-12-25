.. _gmmhmm:

Hidden Markov Model with Gaussian Mixture Model emissions (``GMMHMM``)
======================================================================

The `Hidden Markov Model <https://en.wikipedia.org/wiki/Hidden_Markov_model>`_ (HMM)
is a state-based statistical model that can be used to represent an **individual**
observation sequence class. As seen in the diagram below, the rough idea is that
each state should correspond to one 'section' of the sequence.

.. image:: /_static/hmm.png
    :alt: HMM
    :width: 350
    :align: center
    :name: hmm-img

In the image above, we can imagine creating a HMM to model head gestures of the `'nod'` class.
If the above signal represents the :math:`y`-position of the head during a nod, then
each of the five states above would represent a different 'section' of the nod, and we would fit
this HMM by training it on many :math:`y`-position head gesture signals of the `'nod'` class.

A single HMM is modeled by the :class:`~GMMHMM` class.

Parameters and Training
-----------------------

A HMM is completely determined by its parameters, which are explained below.

- | **Initial state distribution** :math:`\boldsymbol{\pi}`:
  | A probability distribution that dictates the probability of the HMM starting in each state.

- | **Transition probability matrix** :math:`A`:
  | A matrix whose rows represent a probability distribution that dictates how likely the HMM is
    to transition to each state, given some current state.

- | **Emission probability distributions** :math:`B`:
  | A collection of :math:`M` continuous multivariate probability distributions (one for each state)
    that each dictate the probability of the HMM generating an observation :math:`\mathbf{o}`, given some current state.
    Recall that we are generally considering multivariate observation sequences – that is,
    at time :math:`t`, we have an observation :math:`\mathbf{o}^{(t)}=\left(o_1^{(t)}, o_2^{(t)}, \ldots, o_D^{(t)}\right)`.
    The fact that the observations are multivariate necessitates a multivariate emission distribution.
    Sequentia uses a mixture of `multivariate Gaussian distributions <https://en.wikipedia.org/wiki/Multivariate_normal_distribution>`_.

In order to learn these parameters, we must train the HMM on examples that are labeled
with the class :math:`c` that the HMM models. Denote the HMM that models class :math:`c` as
:math:`\lambda_c=(\boldsymbol{\pi}_c, A_c, B_c)`. We can use the `Baum-Welch algorithm <https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm>`_
(an application of the `Expectation-Maximization algorithm <https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm>`_ to HMMs)
to fit :math:`\lambda_c` and learn its parameters. This fitting is implemented by the :func:`~GMMHMM.fit` function.

Mixture Emissions
^^^^^^^^^^^^^^^^^

The assumption that a single multivariate Gaussian emission distribution
is accurate and representative enough to model the probability of observation
vectors of any state of a HMM is often a very strong and naive one. Instead,
a more powerful approach is to represent the emission distribution as
a mixture of multiple multivariate Gaussian densities. An emission distribution
for state :math:`m`, formed by a weighted mixture of :math:`K` multivariate Gaussian densities is defined as:

.. math::
    b_m(\mathbf{o}^{(t)}) = \sum_{k=1}^K c_k^{(m)} \mathcal{N}\big(\mathbf{o}^{(t)}\ ;\ \boldsymbol\mu_k^{(m)}, \Sigma_k^{(m)}\big)

where :math:`\mathbf{o}^{(t)}` is an observation vector at time :math:`t`,
:math:`c_k^{(m)}` is a *mixture weight* such that :math:`\sum_{k=1}^K c_k^{(m)} = 1`
and :math:`\boldsymbol\mu_k^{(m)}` and :math:`\Sigma_k^{(m)}` are the mean vector
and covariance matrix of the :math:`k^\text{th}` mixture component of the :math:`m^\text{th}`
state, respectively.

Note that even in the case that multiple Gaussian densities are not needed, the mixture weights
can be adjusted so that irrelevant Gaussians are omitted and only a single Gaussian remains.
However, the default setting of the :class:`~GMMHMM` class is a single Gaussian.

Then a GMM-HMM is completely determined by the learnable parameters
:math:`\lambda=(\boldsymbol{\pi}, A, B)` where :math:`B=(C,\Pi,\Psi)` and

- :math:`C=\big(c_1^{(m)}, \ldots, c_K^{(m)}\big)_{m=1}^M` is
  a collection of the mixture weights,
- :math:`\Pi=\big(\boldsymbol\mu_1^{(m)}, \ldots, \boldsymbol\mu_K^{(m)}\big)_{m=1}^M` is
  a collection of the mean vectors,
- :math:`\Psi=\big(\Sigma_1^{(m)}, \ldots, \Sigma_K^{(m)}\big)_{m=1}^M` is
  a collection of the covariance matrices,

for every mixture component of each state of the HMM.

Usually if :math:`K` is large enough, a mixture of :math:`K` Gaussian densities can effectively
model any probability density function. With large enough :math:`K`, we can also restrict the
covariance matrices and still get good approximations of any probability density function,
and at the same time decrease the number of parameters that need to be updated during Baum-Welch.

The covariance matrix type can be specified by a string parameter ``covariance_type`` in the
:class:`~GMMHMM` constructor that takes values `'spherical'`, `'diag'`, `'full'` or `'tied'`.
The various types are explained well in this `StackExchange answer <https://stats.stackexchange.com/a/326678>`_,
and summarized in the below image (also courtesy of the same StackExchange answerer).

.. image:: /_static/covariance_types.png
    :alt: Covariance Types
    :width: 100%

Model Topologies
^^^^^^^^^^^^^^^^

As we usually wish to preserve the natural ordering of time, we normally want to prevent our HMM
from transitioning to previous states (this is shown in the `previous figure <#hmm-img>`_). This restriction leads
to what known as a **left-right** HMM, and is the most commonly used type of HMM for sequential
modeling. Mathematically, a left-right HMM is defined by an upper-triangular transition matrix.

A **linear** topology is one in which transitions are only permitted to the current state
and the next state, i.e. no state-jumping is permitted.

If we allow transitions to any state at any time, this HMM topology is known as **ergodic**.

**Note**: Ergodicity is mathematically defined as having a transition matrix with no zero entries.
Using the ergodic topology in Sequentia will still permit zero entries in the transition matrix,
but will issue a warning stating that those probabilities will not be learned.

Sequentia offers all three topologies, specified by a string parameter ``topology`` in the
:class:`~GMMHMM` constructor that takes values `'ergodic'`, `'left-right'` or `'linear'`.

.. image:: /_static/topologies.svg
    :alt: HMM Topologies
    :width: 100%

Making Predictions
------------------

A score for how likely a HMM is to generate an observation sequence is given by the
`Forward algorithm <https://en.wikipedia.org/wiki/Forward_algorithm>`_. It calculates the likelihood
:math:`\mathbb{P}(O|\lambda_c)` of the HMM :math:`\lambda_c` generating the observation sequence :math:`O`.

**Note**: The likelihood does not account for the fact that a particular observation class
may occur more or less frequently than other observation classes. Once a group of :class:`~GMMHMM` objects
(represented by a :class:`~HMMClassifier`) is created and configured, this can be accounted for by
calculating the joint probability (or un-normalized posterior)
:math:`\mathbb{P}(O, \lambda_c)=\mathbb{P}(O|\lambda_c)\mathbb{P}(\lambda_c)`
and using this score to classify instead. The addition of the prior term :math:`\mathbb{P}(\lambda_c)`
accounts for some classes occuring more frequently than others.

Example
-------

.. literalinclude:: ../../_includes/examples/classifiers/gmmhmm.py
    :language: python
    :linenos:

For more elaborate examples, please have a look at the
`example notebooks <https://nbviewer.jupyter.org/github/eonu/sequentia/tree/master/notebooks>`_.

API reference
-------------

.. autoclass:: sequentia.classifiers.hmm.GMMHMM
    :members:

Hidden Markov Model Classifier (``HMMClassifier``)
==================================================

Multiple HMMs can be combined to form a multi-class classifier.
To classify a new observation sequence :math:`O'`, this works by:

1. | Creating and training the HMMs :math:`\lambda_1, \lambda_2, \ldots, \lambda_C`.

2. | Calculating the likelihoods :math:`\mathbb{P}(O'|\lambda_1), \mathbb{P}(O'|\lambda_2), \ldots, \mathbb{P}(O'|\lambda_C)` of each model generating :math:`O'`.

3. | Scaling the likelihoods by priors :math:`\mathbb{P}(\lambda_1), \mathbb{P}(\lambda_2), \ldots, \mathbb{P}(\lambda_C)`, producing un-normalized posteriors
    :math:`\mathbb{P}(O'|\lambda_c)\mathbb{P}(\lambda_c)`.

4. | Performing MAP classification by choosing the class represented by the HMM with the highest posterior – that is,
    :math:`c'=\mathop{\arg\max}_{c\in\{1,\ldots,C\}}{\mathbb{P}(O'|\lambda_c)\mathbb{P}(\lambda_c)}`.

These steps are summarized in the diagram below.

.. image:: /_static/classifier.svg
    :alt: HMM Classifier
    :width: 600
    :align: center

Example
-------

.. literalinclude:: ../../_includes/examples/classifiers/hmm_classifier.py
    :language: python
    :linenos:

For more elaborate examples, please have a look at the `example notebooks <https://nbviewer.jupyter.org/github/eonu/sequentia/tree/master/notebooks>`_.

API reference
-------------

.. autoclass:: sequentia.classifiers.hmm.HMMClassifier
    :members: