Hidden Markov Models
====================

.. toctree::
   :titlesonly:

   classifier
   variants/index

----

The `Hidden Markov Model <https://en.wikipedia.org/wiki/Hidden_Markov_model>`__ (HMM) is a state-based statistical model for sequence modelling.

When used for classification, a HMM can be used to represent an **individual** observation sequence class.
For example, if we were recognizing spoken digits from the Free Spoken Digit Dataset, we would train a separate HMM for each digit, 
to recognize observation sequences belonging to that class.

HMMs can be used to classify both categorical and numerical sequences.

.. seealso::

   See [#jurafsky]_ for a detailed introduction to HMMs.

Parameters and training
-----------------------

A HMM is composed of:

- a **Markov chain**, which models the probability of transitioning between hidden states.
- an **emission model**, which models the probability of emitting an observation from a hidden state.

A HMM :math:`\lambda` is defined by the following parameters:

- | **Initial state distribution** :math:`\boldsymbol{\pi}`:
  | A probability distribution that dictates the probability of the HMM starting in each state.

- | **Transition probability matrix** :math:`A`:
  | A matrix whose rows represent a probability distribution that determine how likely the HMM is
    to transition to each state, given some current state.

  .. note::
     
     Sequentia HMMs are time homogeneous.

- | **Emission probability distributions** :math:`B`:
  | A collection of :math:`M` probability distributions (one for each state) that specify the probability of the HMM
    emitting an observation given some current state.
    
   - | For categorical sequences, the emission distribution :math:`b_m(o^{(t)})` at state :math:`m` is a univariate discrete distribution
       of the probability of the observation :math:`o^{(t)}` at time :math:`t` being one of the possible symbols :math:`\mathcal{S}=\{s_0,s_1,\ldots,s_K\}`.

     This collection of state emission distributions can be modelled as an :math:`M \times K` transition matrix over all states and symbols :math:`\mathcal{S}`.
   
   - | For numerical sequences, the emission distribution :math:`b_m(\mathbf{o}^{(t)})` at state :math:`m` is a multivariate continuous distribution
       of the probability of the observation :math:`\mathbf{o}^{(t)}` at time :math:`t`.

     Numerical sequence support in Sequentia assumes unbounded real-valued emissions which are modelled according to a multivariate Gaussian mixture distribution.

HMMs are fitted to observation sequences using the Baum-Welch (or forward-backward) algorithm which learns all of the parameters described above via Expectation-Maximization (EM).

.. _topologies:

Topologies
----------

The nature of the transition matrix determines the **topology** of the HMM.

Three common types of topology used in sequence modelling are **ergodic**, **left-right** and **linear**.

- **Ergodic topology**: All states have a non-zero probability of transitioning to any state.
- **Left-right topology**: States are arranged in a way such that any state may only transition to itself or any state ahead of it, but not to any previous state.
- **Linear topology**: Same as left-right, but states are only permitted to transition to the next state.

Left-right topologies are particularly useful for modelling sequences where ordering must be respected.

.. image:: /_static/images/topologies.svg
    :alt: HMM Topologies
    :width: 100%

.. note::

   | Sequentia will still permit zero probabilities in an ergodic transition matrix, but will issue a warning stating that these probabilities will not be learned.

Making predictions
------------------

Multiple HMMs trained to recognize individual observation sequence classes can be combined to form a single multi-class classifier that makes predictions according to posterior maximization.

See :ref:`hmm_classifier` for more detail on how HMMs can be used for classification.

.. rubric:: References

.. [#jurafsky] `Daniel Jurafsky and James H. Martin (2009). Speech and Language Processing, Appendix A: Hidden Markov Models, pages 548-563. 2nd Edition. Prentice-Hall, Inc., Upper Saddle River, NJ, USA. <https://web.stanford.edu/~jurafsky/slp3/A.pdf>`_
