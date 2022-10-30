Gaussian Mixture HMM
====================

The Gaussian Mixture HMM is a variant of HMM that uses a multivariate Gaussian mixture model as the emission distribution for each state.

This HMM variant can be used to recognize unbounded real-valued univariate or multivariate sequences.

Emissions
---------

The emission distribution :math:`b_m` of an observation :math:`\mathbf{o}^{(t)}` at time :math:`t` for state :math:`m` is formed by a weighted mixture of :math:`K` multivariate Gaussian probability density functions, defined as:

.. math::

   b_m(\mathbf{o}^{(t)}) = \sum_{k=1}^K c_k^{(m)} \mathcal{N}_D\big(\mathbf{o}^{(t)}\ \big|\ \boldsymbol\mu_k^{(m)}, \Sigma_k^{(m)}\big)

Where:

- | :math:`\mathbf{o}^{(t)}=\left(o_1^{(t)}, o_2^{(t)}, \ldots, o_D^{(t)}\right)` is a single **observation** at time :math:`t`, such that :math:`\mathbf{o}^{(t)}\in\mathbb{R}^D`.
- | :math:`c_k^{(m)}` is a **component mixture weight** for the :math:`k^\text{th}` Gaussian component of the :math:`m^\text{th}` state, such that :math:`\sum_{k=1}^K c_k^{(m)} = 1` and :math:`c_k^{(m)}\in[0, 1]`.
- | :math:`\boldsymbol\mu_k^{(m)}` is a **mean vector** for the :math:`k^\text{th}` Gaussian component of the :math:`m^\text{th}` state, such that :math:`\boldsymbol\mu_k^{(m)}\in\mathbb{R}^D`.
- | :math:`\Sigma_k^{(m)}` is a **covariance matrix** for the :math:`k^\text{th}` Gaussian component of the :math:`m^\text{th}` state, such that :math:`\Sigma_k^{(m)}\in\mathbb{R}^{D\times D}` and :math:`\Sigma_k^{(m)}` is symmetric and positive semi-definite.
- | :math:`\mathcal{N}_D` is the :math:`D`-dimensional multivariate Gaussian probability density function.

Using a mixture rather than a single Gaussian allows for more flexible modelling of observations.

The component mixture weights, mean vector and covariance matrix for all states and Gaussian components are updated during training via Expectation-Maximization through the Baum-Welch algorithm.

.. _covariance_types:

Covariance matrix types
-----------------------

The :math:`K` covariance matrices for a state can come in different forms:

- **Full**: All values are fully learnable independently for each component.
- **Diagonal**: Only values along the diagonal may be learned independently for each component.
- **Spherical**: Same as *diagonal*, with a single value shared along the diagonal for each component.
- **Tied**: Same as *full*, with all components sharing the same single covariance matrix.

Estimating a full covariance matrix is not always necessary, particularly when a sufficient number of Gaussian components are used. If time is limiting, a spherical, diagonal and tied covariance matrix may also yield strong results while reducing training time due to having fewer parameters to estimate.

.. image:: /_static/images/covariance_types.png
   :alt: Covariance Types
   :width: 100%

API reference
-------------

.. autosummary::

   ~sequentia.models.hmm.variants.GaussianMixtureHMM
   ~sequentia.models.hmm.variants.GaussianMixtureHMM.__init__
   ~sequentia.models.hmm.variants.GaussianMixtureHMM.aic
   ~sequentia.models.hmm.variants.GaussianMixtureHMM.bic
   ~sequentia.models.hmm.variants.GaussianMixtureHMM.fit
   ~sequentia.models.hmm.variants.GaussianMixtureHMM.freeze
   ~sequentia.models.hmm.variants.GaussianMixtureHMM.n_params
   ~sequentia.models.hmm.variants.GaussianMixtureHMM.score
   ~sequentia.models.hmm.variants.GaussianMixtureHMM.set_start_probs
   ~sequentia.models.hmm.variants.GaussianMixtureHMM.set_state_covariances
   ~sequentia.models.hmm.variants.GaussianMixtureHMM.set_state_means
   ~sequentia.models.hmm.variants.GaussianMixtureHMM.set_state_weights
   ~sequentia.models.hmm.variants.GaussianMixtureHMM.set_transitions
   ~sequentia.models.hmm.variants.GaussianMixtureHMM.unfreeze

.. autoclass:: sequentia.models.hmm.variants.GaussianMixtureHMM
   :members:
   :inherited-members:
   :exclude-members: get_params, set_params