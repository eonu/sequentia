import numpy as np
from warnings import warn
from .topology import Topology

class ErgodicTopology(Topology):
    """Represents the topology for an ergodic HMM, imposing non-zero probabilities in the transition matrix."""

    def __init__(self, n_states: int, random_state: np.random.RandomState):
        """Parameters:
            n_states {int} - Number of states in the HMM.
            random_state {numpy.random.RandomState} - The random state object for reproducible randomness.
        """
        super().__init__(n_states, random_state)

    def uniform_transitions(self) -> np.ndarray:
        return np.ones((self._n_states, self._n_states)) / self._n_states

    def random_transitions(self) -> np.ndarray:
        return self._random_state.dirichlet(np.ones(self._n_states), size=self._n_states)

    def validate_transitions(self, transitions: np.ndarray) -> None:
        super().validate_transitions(transitions)
        if not np.all(transitions > 0):
            warn('Zero probabilities in ergodic transition matrix - these transition probabilities will not be learned')