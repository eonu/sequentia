import numpy as np, pomegranate as pg, json
from .hmm import HMM

class GMMHMM(HMM):
    """A hidden Markov model representing an isolated temporal sequence class,
    with mixtures of multivariate Gaussian components representing state emission distributions.

    Parameters
    ----------
    label: str
        A label for the model, corresponding to the class being represented.

    n_states: int
        The number of states for the model.

    n_components: int
        The number of mixture components used in the emission distribution for each state.

    covariance: {'diagonal', 'full'}
        The covariance matrix type.

    topology: {'ergodic', 'left-right', 'strict-left-right'}
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

    def __init__(self, label, n_states, n_components, covariance='diagonal', topology='left-right', random_state=None):
        super().__init__(label, n_states, topology, random_state)
        self._n_components = self._val.restricted_integer(
            n_components, lambda x: x > 1, desc='number of mixture components', expected='greater than one')
        self._covariance = self._val.one_of(covariance, ['diagonal', 'full'], desc='covariance matrix type')

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

        # Create a mixture distribution of multivariate Gaussian emission components using combined samples for initial parameter estimation
        concat = np.concatenate(X)
        if self._covariance == 'diagonal':
            # Use diagonal covariance matrices
            dist = pg.GeneralMixtureModel(
                [pg.MultivariateGaussianDistribution(concat.mean(axis=0), concat.std(axis=0) * np.eye(self._n_features)) for _ in range(self._n_components)],
                self._random_state.dirichlet(np.ones(self._n_components)
            )
        )
        else:
            # Use full covariance matrices
            dist = pg.GeneralMixtureModel.from_samples(pg.MultivariateGaussianDistribution, self._n_components, concat)

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

    @property
    def n_components(self):
        return self._n_components

    @property
    def covariance(self):
        return self._covariance

    def as_dict(self):
        """Serializes the :class:`GMMHMM` object into a `dict`, ready to be stored in JSON format.

        Returns
        -------
        serialized: dict
            JSON-ready serialization of the :class:`GMMHMM` object.
        """

        try:
            self._model
        except AttributeError as e:
            raise AttributeError('The model needs to be fitted before it can be exported to a dict') from e

        model = self._model.to_json()

        if 'NaN' in model:
            raise ValueError('Encountered NaN value(s) in HMM parameters')
        else:
            return {
                'type': 'GMMHMM',
                'label': self._label,
                'n_states': self._n_states,
                'n_components': self._n_components,
                'covariance': self._covariance,
                'topology': self._topologies[self._topology.__class__],
                'model': {
                    'initial': self._initial.tolist(),
                    'transitions': self._transitions.tolist(),
                    'n_seqs': self._n_seqs,
                    'n_features': self._n_features,
                    'hmm': json.loads(model)
                }
            }

    @classmethod
    def load(cls, data, random_state=None):
        """Deserializes either a `dict` or JSON serialized :class:`GMMHMM` object.

        Parameters
        ----------
        data: str or dict
            - File path of the serialized JSON data generated by the :meth:`save` method.
            - `dict` representation of the :class:`GMMHMM`, generated by the :meth:`as_dict` method.

        random_state: numpy.random.RandomState, int, optional
            A random state object or seed for reproducible randomness.

        Returns
        -------
        deserialized: :class:`GMMHMM`
            The deserialized HMM object.

        See Also
        --------
        save: Serializes a :class:`GMMHMM` into a JSON file.
        as_dict: Generates a `dict` representation of the :class:`GMMHMM`.
        """

        # Load the serialized GMM-HMM data
        if isinstance(data, dict):
            pass
        elif isinstance(data, str):
            with open(data, 'r') as f:
                data = json.load(f)
        else:
            pass

        # Check that JSON is in the "correct" format
        if data['type'] == 'HMM':
            raise ValueError('You must use the HMM class to deserialize a stored HMM model')
        elif data['type'] == 'GMMHMM':
            pass
        else:
            raise ValueError("Attempted to deserialize an invalid model - expected 'type' field to be 'GMMHMM'")

        # Deserialize the data into a GMM-HMM object
        gmmhmm = cls(data['label'], data['n_states'], data['n_components'], data['covariance'], data['topology'], random_state=random_state)
        gmmhmm._initial = np.array(data['model']['initial'])
        gmmhmm._transitions = np.array(data['model']['transitions'])
        gmmhmm._n_seqs = data['model']['n_seqs']
        gmmhmm._n_features = data['model']['n_features']
        gmmhmm._model = pg.HiddenMarkovModel.from_json(json.dumps(data['model']['hmm']))

        return gmmhmm