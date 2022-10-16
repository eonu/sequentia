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

A HMM is defined by the following parameters:

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

HMMs are fitted to observation sequences using the Baum-Welch (or forward-backward) algorithm.

..   | A collection of :math:`M` continuous multivariate probability distributions (one for each state)
..     that each dictate the probability of the HMM generating an observation :math:`\mathbf{o}`, given some current state.
..     Recall that we are generally considering multivariate observation sequences – that is,
..     at time :math:`t`, we have an observation :math:`\mathbf{o}^{(t)}=\left(o_1^{(t)}, o_2^{(t)}, \ldots, o_D^{(t)}\right)`.
..     The fact that the observations are multivariate necessitates a multivariate emission distribution.
..     Sequentia uses a mixture of `multivariate Gaussian distributions <https://en.wikipedia.org/wiki/Multivariate_normal_distribution>`_.


.. _topologies:

Topologies
----------

TODO

Making predictions
------------------

TODO

.. rubric:: References

.. [#jurafsky] `Daniel Jurafsky and James H. Martin (2009). Speech and Language Processing, Appendix A: Hidden Markov Models, pages 548-563. 2nd Edition. Prentice-Hall, Inc., Upper Saddle River, NJ, USA. <https://web.stanford.edu/~jurafsky/slp3/A.pdf>`_
