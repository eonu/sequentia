import re, numpy as np, hmmlearn.hmm
from .topologies.ergodic import _ErgodicTopology
from .topologies.left_right import _LeftRightTopology
from .topologies.linear import _LinearTopology
from ...internals import _Validator

class GMMHMM:
    """A hidden Markov model (with Gaussian Mixture Model emissions)
    representing a single isolated sequence class.

    Parameters
    ----------
    label: str or numeric
        A label for the model, corresponding to the class being represented.

    n_states: int > 0
        The number of states for the model.

    n_components: int > 0
        The number of mixture components used in the emission distribution for each state.

    covariance_type: {'spherical', 'diag', 'full', 'tied'}
        The covariance matrix type for emission distributions.

    topology: {'ergodic', 'left-right', 'linear'}
        The topology for the model.

    random_state: numpy.random.RandomState, int, optional
        A random state object or seed for reproducible randomness.

    Attributes
    ----------
    label (property): str or numeric
        The label for the model.

    model (property): hmmlearn.hmm.GMMHMM
        The underlying GMMHMM model from `hmmlearn <https://hmmlearn.readthedocs.io/en/latest/api.html#gmmhmm>`_.

    n_states (property): int
        The number of states for the model.

    n_components (property): int
        The number of mixture components used in the emission distribution for each state.

    covariance_type (property): str
        The covariance matrix type for emission distributions.

    frozen (property): set (str)
        The frozen parameters of the HMM or its GMM emission distributions (see :func:`freeze`).

    n_seqs_ (property): int
        The number of observation sequences used to train the model.

    initial_ (property/setter): numpy.ndarray (float)
        The initial state distribution of the model.

    transitions_ (property/setter): numpy.ndarray (float)
        The transition matrix of the model.

    weights_ (property): numpy.ndarray (float)
        The mixture weights of the GMM emission distributions.

    means_ (property): numpy.ndarray (float)
        The mean vectors of the GMM emission distributions.

    covars_ (property): numpy.ndarray (float)
        The covariance matrices of the GMM emission distributions.
    """

    def __init__(self, label, n_states, n_components=1, covariance_type='full', topology='left-right', random_state=None):
        self._val = _Validator()
        self._label = self._val.string_or_numeric(label, 'model label')
        self._label = label
        self._n_states = self._val.restricted_integer(
            n_states, lambda x: x > 0, desc='number of states', expected='greater than zero')
        self._n_components = self._val.restricted_integer(
            n_components, lambda x: x > 0, desc='number of mixture components', expected='greater than zero')
        self._covariance_type = self._val.one_of(covariance_type, ['spherical', 'diag', 'full', 'tied'], desc='covariance matrix type')
        self._val.one_of(topology, ['ergodic', 'left-right', 'linear'], desc='topology')
        self._random_state = self._val.random_state(random_state)
        self._topology = {
            'ergodic': _ErgodicTopology,
            'left-right': _LeftRightTopology,
            'linear': _LinearTopology
        }[topology](self._n_states, self._random_state)
        self._frozen = set()

    def set_uniform_initial(self):
        """Sets a uniform initial state distribution :math:`\\boldsymbol{\\pi}=(\\pi_1,\\pi_2,\\ldots,\\pi_M)` where :math:`\\pi_i=1/M\\quad\\forall i`."""
        self._initial_ = self._topology.uniform_initial()

    def set_random_initial(self):
        """Sets a random initial state distribution by sampling :math:`\\boldsymbol{\\pi}\\sim\\mathrm{Dir}(\\mathbf{1}_M)` where

        - :math:`\\boldsymbol{\\pi}=(\\pi_1,\\pi_2,\\ldots,\\pi_M)` are the initial state probabilities for each state,
        - :math:`\\mathbf{1}_M` is a vector of :math:`M` ones which are used as the concentration parameters for the Dirichlet distribution.
        """
        self._initial_ = self._topology.random_initial()

    def set_uniform_transitions(self):
        """Sets a uniform transition matrix according to the topology, so that given the HMM is in state :math:`i`,
        all permissible transitions (i.e. such that :math:`p_{ij}\\neq0`) :math:`\\forall j` are equally probable."""
        self._transitions_ = self._topology.uniform_transitions()

    def set_random_transitions(self):
        """Sets a random transition matrix according to the topology, so that given the HMM is in state :math:`i`,
        all out-going transition probabilities :math:`\\mathbf{p}_i=(p_{i1},p_{i2},\\ldots,p_{iM})` from state :math:`i`
        are generated by sampling :math:`\\mathbf{p}_i\\sim\\mathrm{Dir}(\\mathbf{1})` with a vector of ones of appropriate
        size used as concentration parameters, so that only transitions permitted by the topology are non-zero."""
        self._transitions_ = self._topology.random_transitions()

    def fit(self, X):
        """Fits the HMM to observation sequences assumed to be labeled as the class that the model represents.

        Parameters
        ----------
        X: list of numpy.ndarray (float)
            Collection of multivariate observation sequences, each of shape :math:`(T \\times D)` where
            :math:`T` may vary per observation sequence.
        """
        (self.initial_, self.transitions_)
        X = self._val.observation_sequences(X)

        # Store the number of sequences and features used to fit the model
        self._n_seqs_, self._n_features_ = len(X), X[0].shape[1]

        # Initialize the GMMHMM with the specified initial state distribution and transition matrix
        self._model = hmmlearn.hmm.GMMHMM(
            n_components=self._n_states,
            n_mix=self._n_components,
            covariance_type=self._covariance_type,
            random_state=self._random_state,
            init_params='mcw', # only initialize means, covariances and mixture weights
            params=(set('stmcw') - self._frozen)
        )
        self._model.startprob_, self._model.transmat_ = self._initial_, self._transitions_

        # Perform the Baum-Welch algorithm to fit the model to the observations
        self._model.fit(np.vstack(X), [len(x) for x in X])

        # Update the initial state distribution and transitions to reflect the updated parameters
        self._initial_, self._transitions_ = self._model.startprob_, self._model.transmat_

    def forward(self, x):
        """Runs the forward algorithm to calculate the (log) likelihood of the model generating an observation sequence.

        Parameters
        ----------
        x: numpy.ndarray (float)
            An individual sequence of observations of size :math:`(T \\times D)` where
            :math:`T` is the number of time frames (or observations) and
            :math:`D` is the number of features.

        Returns
        -------
        log-likelihood: float
            The log-likelihood of the model generating the observation sequence.
        """
        self.model
        x = self._val.observation_sequences(x, allow_single=True)
        if not x.shape[1] == self._n_features_:
            raise ValueError('Number of observation features must match the dimensionality of the original data used to fit the model')
        return self._model.score(x, lengths=None)

    def freeze(self, params=None):
        """Freezes the specified parameters of the HMM or its GMM emission distributions,
        preventing them from being updated during the Baum–Welch algorithm.

        Parameters
        ----------
        params: str, optional
            | A string specifying which parameters to freeze.
            | Can contain any combination of:

            - `'s'` for initial state probabilities (HMM parameters),
            - `'t'` for transition probabilities (HMM parameters),
            - `'m'` for mean vectors (GMM emission distribution parameters),
            - `'c'` for covariance matrices (GMM emission distribution parameters),
            - '`w`' for mixing weights (GMM emission distribution parameters).

            Defaults to all parameters, i.e. `'stmcw'`.

        See Also
        --------
        unfreeze : Unfreezes parameters of the HMM or its GMM emission distributions.
        """
        self._frozen |= set(self._modify_params(params))

    def unfreeze(self, params=None):
        """Unfreezes the specified parameters of the HMM or its GMM emission distributions
        which were frozen with :func:`freeze`, allowing them to be updated during the Baum–Welch algorithm.

        Parameters
        ----------
        params: str, optional
            | A string specifying which parameters to unfreeze.
            | Can contain any combination of:

            - `'s'` for initial state probabilities (HMM parameters),
            - `'t'` for transition probabilities (HMM parameters),
            - `'m'` for mean vectors (GMM emission distribution parameters),
            - `'c'` for covariance matrices (GMM emission distribution parameters),
            - '`w`' for mixing weights (GMM emission distribution parameters).

            Defaults to all parameters, i.e. `'stmcw'`.

        See Also
        --------
        freeze : Freezes parameters of the HMM or its GMM emission distributions.
        """
        self._frozen -= set(self._modify_params(params))

    def _modify_params(self, params):
        if isinstance(params, str):
            if bool(re.compile(r'[^stmcw]').search(params)):
                raise ValueError("Expected a string consisting of any combination of 's', 't', 'm', 'c', 'w'")
        elif params is None:
            params = 'stmcw'
        else:
            raise TypeError("Expected a string consisting of any combination of 's', 't', 'm', 'c', 'w'")
        return params

    @property
    def label(self):
        return self._label

    @property
    def n_states(self):
        return self._n_states

    @property
    def n_components(self):
        return self._n_components

    @property
    def covariance_type(self):
        return self._covariance_type

    @property
    def n_seqs_(self):
        return self._val.fitted(self,
            lambda self: self._n_seqs_,
            'The model has not been fitted and has not seen any observation sequences'
        )

    @property
    def frozen(self):
        return self._frozen

    @property
    def model(self):
        return self._val.fitted(self,
            lambda self: self._model,
            'The model must be fitted first'
        )

    @property
    def initial_(self):
        return self._val.fitted(self,
            lambda self: self._initial_,
            'No initial state distribution has been defined'
        )

    @initial_.setter
    def initial_(self, probabilities):
        self._topology.validate_initial(probabilities)
        self._initial_ = probabilities

    @property
    def transitions_(self):
        return self._val.fitted(self,
            lambda self: self._transitions_,
            'No transition matrix has been defined'
        )

    @transitions_.setter
    def transitions_(self, probabilities):
        self._topology.validate_transitions(probabilities)
        self._transitions_ = probabilities

    @property
    def weights_(self):
        return self.model.weights_

    @property
    def means_(self):
        return self.model.means_

    @property
    def covars_(self):
        return self.model.covars_

    def __repr__(self):
        name = '.'.join([self.__class__.__module__.split('.')[0], self.__class__.__name__])
        attrs = [
            ('label', repr(self._label)),
            ('n_states', repr(self._n_states)),
            ('n_components', repr(self._n_components)),
            ('covariance_type', repr(self._covariance_type)),
            ('frozen', repr(self._frozen))
        ]
        return '{}({})'.format(name, ', '.join('{}={}'.format(name, val) for name, val in attrs))