import numpy as np
import pomegranate as pg
from .topologies.ergodic import ErgodicTopology
from .topologies.left_right import LeftRightTopology
from typing import List

class HMM:
    """A hidden Markov model representing an isolated temporal sequence.

    Example:
        >>> import numpy as np
        >>> from sequentia.classifiers import HMM
        >>>
        >>> X = [np.random.random((10 * i, 3)) for i in range(1, 4)]
        >>>
        >>> nod = HMM(label='nod', n_states=5, topology='left-right')
        >>> nod.set_random_initial()
        >>> nod.set_random_transitions()
        >>> nod.fit(X)

    Attributes:
        label (getter) - The label for the model.
        n_states (getter) - The number of states for the model.
        prior (getter) - The prior for the model.
        n_seqs (getter) - The number of observation sequences use to train the model.
        initial (setter/getter) - The initial state distribution of the model.
        transitions (setter/getter) - The transition matrix of the model.
    """

    def __init__(self, label: str, n_states: int, topology='ergodic', prior=None, random_state=None):
        """
        Parameters:
            label {str} - A label for the model (should ideally correspond to the class label).
            n_states {int} - The number of states for the model.
            topology {str} - The topology ('ergodic' or 'left-right') for the model.
            prior {float} - The prior for the class represented by the model.
            random_state {numpy.random.RandomState, int} - A random state object or seed for reproducible randomness.
        """
        if not isinstance(label, str):
            raise TypeError('Expected `label` to be a string')
        self._label = label

        if not isinstance(n_states, int):
            raise TypeError('Expected `n_states` to be an int')
        if n_states < 1:
            raise ValueError('A HMM requires at least one state')
        self._n_states = n_states

        if prior is not None:
            if not isinstance(prior, float):
                raise TypeError('Model prior probability must be a float.')
            if not 0 < prior <= 1:
                raise ValueError('Model prior probability must be greater than 0 and less than or equal to 1')
        self._prior = prior

        if random_state is None:
            self._random_state = np.random.RandomState()
        elif isinstance(random_state, np.random.RandomState):
            self._random_state = random_state
        elif isinstance(random_state, int):
            self._random_state = np.random.RandomState(seed=random_state)
        else:
            raise TypeError("Expected `random_state` to be of type: None, int, or numpy.random.RandomState")

        if topology == 'ergodic':
            self._topology = ErgodicTopology(self._n_states, self._random_state)
        elif topology == 'left-right':
            self._topology = LeftRightTopology(self._n_states, self._random_state)
        else:
            raise ValueError("Expected `topology` to be: 'ergodic' or 'left-right'")

    @property
    def label(self) -> str:
        return self._label

    @property
    def n_states(self) -> int:
        return self._n_states

    @property
    def prior(self) -> float:
        return self._prior

    @property
    def n_seqs(self) -> int:
        """Number of observation sequences used to train the model."""
        try:
            return self._n_seqs
        except AttributeError as e:
            raise AttributeError('The model has not been fitted and has not seen any observation sequences') from e

    @property
    def initial(self) -> np.ndarray:
        try:
            return self._initial
        except AttributeError as e:
            raise AttributeError('No initial state distribution has been defined') from e

    @initial.setter
    def initial(self, probabilities: np.ndarray):
        self._topology.validate_initial(probabilities)
        self._initial = probabilities

    @property
    def transitions(self) -> np.ndarray:
        try:
            return self._transitions
        except AttributeError as e:
            raise AttributeError('No transition matrix has been defined') from e

    @transitions.setter
    def transitions(self, probabilities: np.ndarray):
        self._topology.validate_transitions(probabilities)
        self._transitions = probabilities

    def set_uniform_initial(self):
        self._initial = self._topology.uniform_initial()

    def set_random_initial(self):
        self._initial = self._topology.random_initial()

    def set_uniform_transitions(self):
        self._transitions = self._topology.uniform_transitions()

    def set_random_transitions(self):
        self._transitions = self._topology.random_transitions()

    def fit(self, X: List[np.ndarray]):
        """Fits the HMM to observation sequences assumed to be labeled as the class that the model represents.

        Parameters:
            X {list(numpy.ndarray)} - Collection of multivariate observation sequences, each of shape (T, D)
                where T may vary per observation sequence.
        """
        if not isinstance(X, list):
            raise TypeError('Collection of observation sequences must be a list')
        if not all(isinstance(sequence, np.ndarray) for sequence in X):
            raise TypeError('Each observation sequence must be a numpy.ndarray')
        if not all(sequence.ndim == 2 for sequence in X):
            raise ValueError('Each observation sequence must be two-dimensional')
        if not all(sequence.shape[1] == X[0].shape[1] for sequence in X):
            raise ValueError('Each observation sequence must have the same dimensionality')

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
        self._model.fit(X)

        # Update the initial state distribution and transitions to reflect the updated parameters
        inner_tx = self._model.dense_transition_matrix()[:, :self._n_states]
        self._initial = inner_tx[self._n_states]
        self._transitions = inner_tx[:self._n_states]

    def forward(self, sequence: np.ndarray) -> float:
        """Runs the forward algorithm to calculate the (negative log) likelihood of the model generating an observation sequence.

        Parameters:
            sequence {numpy.ndarray} - An individual sequence of observations of size (T, D) where:
                T is the number of time frames (or observations) and D is the number of features.

        Returns {float}:
            The negative log-likelihood of the model generating the observation sequence.
        """
        if not isinstance(sequence, np.ndarray):
            raise TypeError('Sequence of observations must be a numpy.ndarray')
        if not sequence.ndim == 2:
            raise ValueError('Sequence of observations must be two-dimensional')
        if not sequence.shape[1] == self._n_features:
            raise ValueError('Number of observation features must match the dimensionality of the original data used to fit the model')

        return -self._model.log_probability(sequence)