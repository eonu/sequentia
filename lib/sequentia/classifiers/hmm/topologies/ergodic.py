import numpy as np
from warnings import warn
from .topology import _Topology

class _ErgodicTopology(_Topology):
    """Represents the topology for an ergodic HMM, imposing non-zero probabilities in the transition matrix.

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
        return np.ones((self._n_states, self._n_states)) / self._n_states

    def random_transitions(self) -> np.ndarray:
        """Sets the transition matrix as random (random probability of transitioning
        to all other possible states from each state) by sampling probabilities
        from a Dirichlet distribution - according to the topology.

        Returns
        -------
        transitions: :class:`numpy:numpy.ndarray` (float)
            The random transition matrix of shape `(n_states, n_states)`.
        """
        return self._random_state.dirichlet(np.ones(self._n_states), size=self._n_states)

    def validate_transitions(self, transitions: np.ndarray) -> None:
        """Validates a transition matrix according to the topology's restrictions.

        Parameters
        ----------
        transitions: numpy.ndarray (float)
            The transition matrix to validate.
        """
        super().validate_transitions(transitions)
        if not np.all(transitions > 0):
            warn('Zero probabilities in ergodic transition matrix - these transition probabilities will not be learned')