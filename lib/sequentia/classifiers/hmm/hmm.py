import numpy as np
import pomegranate as pg
from .topologies.ergodic import _ErgodicTopology
from .topologies.left_right import _LeftRightTopology
from ...internals import _Validator

class HMM:
    """A hidden Markov model representing an isolated temporal sequence class.

    Parameters
    ----------
    label: str
        A label for the model, corresponding to the class being represented.

    n_states: int
        The number of states for the model.

    topology: {'ergodic', 'left-right'}
        The topology for the model.

    random_state: numpy.random.RandomState, int, optional
        A random state object or seed for reproducible randomness.

    Attributes
    ----------
    label: str
        The label for the model.

    n_states: int
        The number of states for the model.

    n_seqs: int
        The number of observation sequences use to train the model.

    initial: numpy.ndarray
        The initial state distribution of the model.

    transitions: numpy.ndarray
        The transition matrix of the model.
    """

    def __init__(self, label, n_states, topology='left-right', random_state=None):
        self._val = _Validator()
        self._label = self._val.string(label, 'model label')
        self._n_states = self._val.restricted_integer(
            n_states, lambda x: x > 0, desc='number of states', expected='greater than zero')
        self._val.one_of(topology, ['ergodic', 'left-right'], desc='topology')
        self._random_state = self._val.random_state(random_state)

        if topology == 'ergodic':
            self._topology = _ErgodicTopology(self._n_states, self._random_state)
        elif topology == 'left-right':
            self._topology = _LeftRightTopology(self._n_states, self._random_state)

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

    def fit(self, X, n_jobs=1):
        """Fits the HMM to observation sequences assumed to be labeled as the class that the model represents.

        Parameters
        ----------
        X: List[numpy.ndarray]
            Collection of multivariate observation sequences, each of shape :math:`(T \\times D)` where
            :math:`T` may vary per observation sequence.

        n_jobs: int
            | The number of jobs to run in parallel.
            | Setting this to -1 will use all available CPU cores.
        """
        X = self._val.observation_sequences(X)
        self._val.restricted_integer(n_jobs, lambda x: x == -1 or x > 0, 'number of jobs', '-1 or greater than zero')

        try:
            (self._initial, self._transitions)
        except AttributeError as e:
            raise AttributeError('Must specify initial state distribution and transitions before the HMM can be fitted') from e

        self._n_seqs = len(X)
        self._n_features = X[0].shape[1]

        # Create a multivariate Gaussian emission distribution using combined samples for initial parameter esimation
        dist = pg.MultivariateGaussianDistribution.from_samples(np.concatenate(X))

        # Create the HMM object
        self._model = pg.HiddenMarkovModel.from_matrix(
            name=self._label,
            transition_probabilities=self._transitions,
            distributions=[dist.copy() for _ in range(self._n_states)],
            starts=self._initial
        )

        # Perform the Baum-Welch algorithm to fit the model to the observations
        self._model.fit(X, n_jobs=n_jobs)

        # Update the initial state distribution and transitions to reflect the updated parameters
        inner_tx = self._model.dense_transition_matrix()[:, :self._n_states]
        self._initial = inner_tx[self._n_states]
        self._transitions = inner_tx[:self._n_states]

    def forward(self, sequence):
        """Runs the forward algorithm to calculate the (negative log) likelihood of the model generating an observation sequence.

        Parameters
        ----------
        sequence: numpy.ndarray
            An individual sequence of observations of size :math:`(T \\times D)` where
            :math:`T` is the number of time frames (or observations) and
            :math:`D` is the number of features.

        Returns
        -------
        negative log-likelihood: float
            The negative log-likelihood of the model generating the observation sequence.
        """
        try:
            self._model
        except AttributeError as e:
            raise AttributeError('The model must be fitted before running the forward algorithm') from e

        sequence = self._val.observation_sequences(sequence, allow_single=True)
        if not sequence.shape[1] == self._n_features:
            raise ValueError('Number of observation features must match the dimensionality of the original data used to fit the model')

        return -self._model.log_probability(sequence)

    @property
    def label(self):
        return self._label

    @property
    def n_states(self):
        return self._n_states

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