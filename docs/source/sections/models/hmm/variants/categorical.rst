Categorical HMM
===============

The Categorical HMM is a variant of HMM that uses a discrete probability distribution over a finite set of symbols as the emission distribution for each state.

This HMM variant can be used to recognize categorical univariate sequences.

Emissions
---------

The emission distribution :math:`b_m` of an observation :math:`o^{(t)}` at time :math:`t` for state :math:`m` is given by a probability vector:

.. math::

   \bigg[\underbrace{\mathbb{P}\big(o^{(t)}=s_0\ |\ q^{(t)}=m\big)}_{p_{m,0}}, \ldots, \underbrace{\mathbb{P}\big(o^{(t)}=s_K\ |\ q^{(t)}=m\big)}_{p_{m,K}}\bigg]

Where:

- | :math:`\mathcal{S}=\{s_0,s_1,\ldots,s_K\}` is a finite set of **observation symbols**.
- | :math:`o^{(t)}\in\mathcal{S}` is a single **observation** at time :math:`t`.
- | :math:`q^{(t)}` is a discrete random variable representing the hidden state at time :math:`t`.
- | :math:`p_{m,k}=\mathbb{P}\big(o^{(t)}=s_k\ |\ q^{(t)}=m\big)` is the probability of observing :math:`s_k` while in state :math:`m`.

The emission distributions for all states can be represented by a single :math:`M\times K` **emission matrix**:

.. math::

   \begin{bmatrix}
      p_{0,0} & \cdots & p_{0,K} \\
      \vdots  & \ddots & \vdots \\
      p_{M,0} & \cdots & p_{M,K}
   \end{bmatrix}

.. note::

   Observation symbols must be encoded as integers. Consider performing label encoding using :class:`sklearn:sklearn.preprocessing.LabelEncoder`.

API reference
-------------

Class
^^^^^

.. autosummary::

   ~sequentia.models.hmm.variants.CategoricalHMM

Methods
^^^^^^^

.. autosummary::

   ~sequentia.models.hmm.variants.CategoricalHMM.__init__
   ~sequentia.models.hmm.variants.CategoricalHMM.aic
   ~sequentia.models.hmm.variants.CategoricalHMM.bic
   ~sequentia.models.hmm.variants.CategoricalHMM.fit
   ~sequentia.models.hmm.variants.CategoricalHMM.freeze
   ~sequentia.models.hmm.variants.CategoricalHMM.score
   ~sequentia.models.hmm.variants.CategoricalHMM.set_state_emission_probs
   ~sequentia.models.hmm.variants.CategoricalHMM.set_state_start_probs
   ~sequentia.models.hmm.variants.CategoricalHMM.set_state_transition_probs
   ~sequentia.models.hmm.variants.CategoricalHMM.unfreeze
   ~sequentia.models.hmm.variants.CategoricalHMM.n_params

.. _definitions:

Definitions
^^^^^^^^^^^

.. autoclass:: sequentia.models.hmm.variants.CategoricalHMM
   :members:
   :inherited-members:
   :exclude-members: get_params, set_params, get_metadata_routing, set_fit_request, set_score_request
