import numpy as np, hmmlearn.hmm
from .topologies.ergodic import _ErgodicTopology
from .topologies.left_right import _LeftRightTopology
from .topologies.strict_left_right import _StrictLeftRightTopology
from ...internals import _Validator

class GMMHMM:
    """A hidden Markov model representing an isolated sequence class.

    Parameters
    ----------
    label: str or numeric
        A label for the model, corresponding to the class being represented.

    n_states: int > 0
        The number of states for the model.

    n_components: int > 0
        The number of mixture components used in the emission distribution for each state.

    covariance_type: {'spherical', 'diag', 'full', 'tied'}
        The covariance matrix type.

    topology: {'ergodic', 'left-right', 'strict-left-right'}
        The topology for the model.

    random_state: numpy.random.RandomState, int, optional
        A random state object or seed for reproducible randomness.

    Attributes
    ----------
    label: str or numeric
        The label for the model.

    n_states: int
        The number of states for the model.

    n_seqs: int
        The number of observation sequences use to train the model.

    initial: numpy.ndarray
        The initial state distribution of the model.

    transitions: numpy.ndarray
        The transition matrix of the model.

    TODO: Add all other fields
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
        self._val.one_of(topology, ['ergodic', 'left-right', 'strict-left-right'], desc='topology')
        self._random_state = self._val.random_state(random_state)
        self._topology = {
            'ergodic': _ErgodicTopology,
            'left-right': _LeftRightTopology,
            'strict-left-right': _StrictLeftRightTopology
        }[topology](self._n_states, self._random_state)

    def set_uniform_initial(self):
        """Sets a uniform initial state distribution."""
        self._initial = self._topology.uniform_initial()

    def set_random_initial(self):
        """Sets a random initial state distribution."""
        self._initial = self._topology.random_initial()

    def set_uniform_transitions(self):
        """Sets a uniform transition matrix according to the topology."""
        self._transitions = self._topology.uniform_transitions()

    def set_random_transitions(self):
        """Sets a random transition matrix according to the topology."""
        self._transitions = self._topology.random_transitions()

    def fit(self, X):
        """Fits the HMM to observation sequences assumed to be labeled as the class that the model represents.

        Parameters
        ----------
        X: List[numpy.ndarray]
            Collection of multivariate observation sequences, each of shape :math:`(T \\times D)` where
            :math:`T` may vary per observation sequence.
        """
        X = self._val.observation_sequences(X)

        try:
            (self._initial, self._transitions)
        except AttributeError as e:
            raise AttributeError('Must specify initial state distribution and transitions before the HMM can be fitted') from e

        self._n_seqs = len(X)
        self._n_features = X[0].shape[1]

        # Initialize the GMMHMM with the specified initial state distribution and transition matrix
        self._model = hmmlearn.hmm.GMMHMM(
            n_components=self._n_states,
            n_mix=self._n_components,
            covariance_type=self._covariance_type,
            random_state=self._random_state,
            init_params='mcw', # only initialize means, covariances and mixture weights
        )
        self._model.startprob_, self._model.transmat_ = self._initial, self._transitions

        # Perform the Baum-Welch algorithm to fit the model to the observations
        self._model.fit(np.vstack(X), [len(x) for x in X])

        # Update the initial state distribution and transitions to reflect the updated parameters
        self._initial, self._transitions = self._model.startprob_, self._model.transmat_

    def forward(self, x):
        """Runs the forward algorithm to calculate the (log) likelihood of the model generating an observation sequence.

        Parameters
        ----------
        x: numpy.ndarray
            An individual sequence of observations of size :math:`(T \\times D)` where
            :math:`T` is the number of time frames (or observations) and
            :math:`D` is the number of features.

        Returns
        -------
        log-likelihood: float
            The log-likelihood of the model generating the observation sequence.
        """
        try:
            self._model
        except AttributeError as e:
            raise AttributeError('The model must be fitted before running the forward algorithm') from e

        x = self._val.observation_sequences(x, allow_single=True)
        if not x.shape[1] == self._n_features:
            raise ValueError('Number of observation features must match the dimensionality of the original data used to fit the model')

        return self._model.score(x, lengths=None)

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
    def n_seqs(self):
        try:
            return self._n_seqs
        except AttributeError as e:
            raise AttributeError('The model has not been fitted and has not seen any observation sequences') from e

    @property
    def initial(self):
        try:
            return self._initial
        except AttributeError as e:
            raise AttributeError('No initial state distribution has been defined') from e

    @initial.setter
    def initial(self, probabilities):
        self._topology.validate_initial(probabilities)
        self._initial = probabilities

    @property
    def transitions(self):
        try:
            return self._transitions
        except AttributeError as e:
            raise AttributeError('No transition matrix has been defined') from e

    @transitions.setter
    def transitions(self, probabilities):
        self._topology.validate_transitions(probabilities)
        self._transitions = probabilities