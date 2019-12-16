import numpy as np
from .topology import Topology

class LeftRightTopology(Topology):
    """Represents the topology for a left-right HMM, imposing an upper-triangular transition matrix."""

    def __init__(self, n_states: int, random_state: np.random.RandomState):
        """Parameters:
            n_states {int} - Number of states in the HMM.
            random_state {numpy.random.RandomState} - The random state object for reproducible randomness.
        """
        super().__init__(n_states, random_state)

    def uniform_transitions(self) -> np.ndarray:
        upper_ones = np.triu(np.ones((self._n_states, self._n_states)))
        upper_divisors = np.triu(np.tile(np.arange(self._n_states, 0, -1), (self._n_states, 1)).T)
        lower_ones = np.tril(np.ones(self._n_states), k=-1) # One-pad lower triangle to prevent zero division
        return upper_ones / (upper_divisors + lower_ones)

    def random_transitions(self) -> np.ndarray:
        transitions = self._random_state.dirichlet(np.ones(self._n_states), size=self._n_states)
        lower_sums = np.sum(np.tril(transitions, k=-1), axis=1) # Amount to be redistributed per row
        quantities = np.arange(self._n_states, 0, -1) # Number of elements per row to redistribute evenly to
        upper_ones = np.triu(np.ones((self._n_states, self._n_states)))
        redist = (lower_sums / quantities).reshape(-1, 1) * upper_ones
        return np.triu(transitions) + redist

    def validate_transitions(self, transitions: np.ndarray) -> None:
        super().validate_transitions(transitions)
        if not np.allclose(transitions, np.triu(transitions)):
            raise ValueError('Left-right transition matrix must be upper-triangular')