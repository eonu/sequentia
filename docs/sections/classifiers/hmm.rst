.. _hmm:

Hidden Markov Model (``HMM``)
=============================

The `Hidden Markov Model <https://en.wikipedia.org/wiki/Hidden_Markov_model>`_ (HMM)
is a state-based statistical model that can be used to represent an individual
observation sequence class :math:`c`. As seen in the diagram below, the rough idea is that
each state should correspond to one 'section' of the sequence.

.. image:: https://i.ibb.co/GFtV46t/HMM.jpg
    :alt: HMM
    :width: 350

A single HMM is modeled by the :class:`~HMM` class.

Parameters and Training
-----------------------

The 'sections' in the image above are determined by the parameters of the HMM, explained below.

- | **Initial state distribution** :math:`\boldsymbol{\pi}`:
  | A discrete probability distribution that dictates the probability of the HMM starting in each state.

- | **Transition probability matrix** :math:`A`:
  | A matrix whose rows represent a discrete probability distribution that dictates how likely the HMM is
    to transition to each state, given some current state.

- | **Emission probability distributions** :math:`B`:
  | A collection of :math:`N` continuous multivariate probability distributions (one for each state)
    that each dictate the probability of the HMM generating an observation :math:`\mathbf{o}`, given some current state.
    Recall that we are generally considering multivariate observation sequences – that is,
    at time :math:`t`, we have an observation :math:`\mathbf{o}^{(t)}=\left(o_1^{(t)}, o_2^{(t)}, \ldots, o_D^{(t)}\right)`.
    The fact that the observations are multivariate necessitates a multivariate emission distribution.
    Sequentia uses the `multivariate Gaussian distribution <https://en.wikipedia.org/wiki/Multivariate_normal_distribution>`_.

In order to learn these parameters, we must train the HMM on examples that are labeled
with the class :math:`c` that the HMM models. Denote the HMM that models class :math:`c` as
:math:`\lambda_c=(\boldsymbol{\pi}_c, A_c, B_c)`. We can use the `Baum-Welch algorithm <https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm>`_
(an application of the `Expectation-Maximization algorithm <https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm>`_)
to fit :math:`\lambda_c` and learn its parameters. This fitting is implemented by the :func:`~HMM.fit` function.

Model Topologies
^^^^^^^^^^^^^^^^

As we usually wish to preserve the natural ordering of time, we normally want to prevent our HMM
from transitioning to previous states (this is shown in the figure above). This restriction leads
to what known as a **left-right** HMM, and is the most commonly used type of HMM for sequential
modeling. Mathematically, a left-right HMM is defined by an upper-triangular transition matrix.

If we allow transitions to any state at any time, this HMM topology is known as **ergodic**.

**Note**: Ergodicity is mathematically defined as having a transition matrix with no non-zero entries.
Using the ergodic topology in Sequentia will still permit zero entries in the transition matrix,
but will issue a warning stating that those probabilities will not be learned.

Sequentia offers both topologies, specified by a string parameter ``topology`` in the
:class:`~HMM` constructor that takes values `'left-right'` or `'ergodic'`.

Making Predictions
------------------

A score for how likely a HMM is to generate an observation sequence is given by the
`Forward algorithm <https://en.wikipedia.org/wiki/Forward_algorithm>`_. It calculates the likelihood
:math:`\mathbb{P}(O|\lambda_c)` of the HMM :math:`\lambda_c` generating the observation sequence :math:`O`.

**Note**: The likelihood does not account for the fact that a particular observation class
may occur more or less frequently than other observation classes. Once a group of :class:`~HMM` objects
(represented by a :class:`~HMMClassifier`) is created and configured, this can be accounted for by
calculating the joint probability (or un-normalized posterior)
:math:`\mathbb{P}(O, \lambda_c)=\mathbb{P}(O|\lambda_c)\mathbb{P}(\lambda_c)`
and using this score to classify instead. The addition of the prior term :math:`\mathbb{P}(\lambda_c)`
accounts for some classes occuring more frequently than others.

Example
-------

.. literalinclude:: ../../_includes/examples/classifiers/hmm.py
    :language: python
    :linenos:

For more elaborate examples, please have a look at the
`example notebooks <https://nbviewer.jupyter.org/github/eonu/sequentia/tree/master/notebooks>`_.

API reference
-------------

.. autoclass:: sequentia.classifiers.hmm.HMM
    :members:

Hidden Markov Model Classifier (``HMMClassifier``)
==================================================

Multiple HMMs can be combined to form a multi-class classifier.
To classify a new observation sequence :math:`O'`, this works by:

1. | Creating and training the HMMs :math:`\lambda_1, \lambda_2, \ldots, \lambda_N`.

2. | Calculating the likelihoods :math:`\mathbb{P}(O'|\lambda_1), \mathbb{P}(O'|\lambda_2), \ldots, \mathbb{P}(O'|\lambda_N)` of each model generating :math:`O'`.
   | **Note**: You can also used the un-normalized posterior :math:`\mathbb{P}(O'|\lambda_c)\mathbb{P}(\lambda_c)` instead of the likelihood.

3. | Choose the class represented by the HMM with the highest likelihood – that is, :math:`c^*=\mathop{\arg\max}_{c\in\{1,\ldots,N\}}{\mathbb{P}(O'|\lambda_c)}`.

These steps are summarized in the diagram below.

.. image:: https://i.ibb.co/gPymgs4/classifier.png
    :alt: HMM Classifier
    :width: 400

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