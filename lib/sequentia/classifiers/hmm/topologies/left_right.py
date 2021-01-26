import numpy as np
from .topology import _Topology

class _LeftRightTopology(_Topology):
    """Represents the topology for a left-right HMM, imposing an upper-triangular transition matrix.

    Parameters
    ----------
    n_states: int
        Number of states in the HMM.

    random_state: numpy.random.RandomState
        A random state object for reproducible randomness.
    """

    def uniform_transitions(self) -> np.ndarray:
        """Sets the transition matrix as uniform (equal probability of transitioning
            to all other possible states from each state) corresponding to the topology.

        Returns
        -------
        transitions: :class:`numpy:numpy.ndarray` (float)
            The uniform transition matrix of shape `(n_states, n_states)`.
        """
        upper_ones = np.triu(np.ones((self._n_states, self._n_states)))
        upper_divisors = np.triu(np.tile(np.arange(self._n_states, 0, -1), (self._n_states, 1)).T)
        lower_ones = np.tril(np.ones(self._n_states), k=-1) # One-pad lower triangle to prevent zero division
        return upper_ones / (upper_divisors + lower_ones)

    def random_transitions(self) -> np.ndarray:
        """Sets the transition matrix as random (random probability of transitioning
        to all other possible states from each state) by sampling probabilities
        from a Dirichlet distribution, according to the topology.

        Returns
        -------
        transitions: :class:`numpy:numpy.ndarray` (float)
            The random transition matrix of shape `(n_states, n_states)`.
        """
        transitions = np.zeros((self._n_states, self._n_states))
        for i, row in enumerate(transitions):
            row[i:] = self._random_state.dirichlet(np.ones(self._n_states - i))
        return transitions

    def validate_transitions(self, transitions: np.ndarray) -> None:
        """Validates a transition matrix according to the topology's restrictions.

        Parameters
        ----------
        transitions: numpy.ndarray (float)
            The transition matrix to validate.
        """
        super().validate_transitions(transitions)
        if not np.allclose(transitions, np.triu(transitions)):
            raise ValueError('Left-right transition matrix must be upper-triangular')