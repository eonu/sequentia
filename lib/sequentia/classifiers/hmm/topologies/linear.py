import numpy as np
from .topology import _Topology

class _LinearTopology(_Topology):
    """Represents the topology for a linear HMM.

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
        transitions = np.zeros((self._n_states, self._n_states))
        for i, row in enumerate(transitions[:-1]):
            row[i:(i+2)] = np.ones(2) / 2
        transitions[self._n_states - 1][self._n_states - 1] = 1
        return transitions

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
        for i, row in enumerate(transitions[:-1]):
            row[i:(i+2)] = self._random_state.dirichlet(np.ones(2))
        transitions[self._n_states - 1][self._n_states - 1] = 1
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
        if not np.allclose(transitions, np.diag(np.diag(transitions)) + np.diag(np.diag(transitions, k=1), k=1)):
            raise ValueError('Linear transition matrix must only consist of a diagonal and upper diagonal')