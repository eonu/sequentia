import numpy as np, json
from .hmm import HMM
from .gmmhmm import GMMHMM
from sklearn.metrics import confusion_matrix
from ...internals import _Validator

class HMMClassifier:
    """A classifier that combines individual :class:`~HMM` and/or :class:`~GMMHMM` objects,
    which model isolated sequences from different classes."""

    def __init__(self):
        self._val = _Validator()

    def fit(self, models):
        """Fits the classifier with a collection of :class:`~HMM` and/or :class:`~GMMHMM` objects.

        Parameters
        ----------
        models: List[HMM, GMMHMM] or Dict[Any, HMM/GMMHMM]
            A collection of :class:`~HMM` objects to use for classification.
        """
        if isinstance(models, list):
            if not all(isinstance(model, HMM) for model in models):
                raise TypeError('Expected all models to be HMM objects')
        elif isinstance(models, dict):
            values = list(models.values())
            if not all(isinstance(model, HMM) for model in values):
                raise TypeError('Expected all models to be HMM objects')
            models = values
        else:
            raise TypeError('Expected models to be a list or dict of HMM objects')

        if len(models) > 0:
            self._models = models
        else:
            raise RuntimeError('Must fit the classifier with at least one HMM')

    def predict(self, X, prior='frequency', return_scores=False):
        """Predicts the label for an observation sequence (or multiple sequences) according to maximum likelihood or posterior scores.

        Parameters
        ----------
        X: numpy.ndarray or List[numpy.ndarray]
            An individual observation sequence or a list of multiple observation sequences.

        prior: {'frequency', 'uniform'} or Dict[str, float]
            How the prior for each model is calculated to perform MAP estimation by scoring with
            the joint probability (or un-normalized posterior) :math:`\mathbb{P}(O, \lambda_c)=\mathbb{P}(O|\lambda_c)\mathbb{P}(\lambda_c)`,
            where the likelihood :math:`\mathbb{P}(O|\lambda_c)` is generated from the models' :func:`~HMM.forward` function.

            - `'frequency'`: Calculate the prior :math:`\mathbb{P}(\lambda_c)` as the proportion of training examples in class :math:`c`.
            - `'uniform'`: Set the priors uniformly such that :math:`\mathbb{P}(\lambda_c)=\\frac{1}{C}` for each class :math:`c\in\{1,\ldots,C\}`.

            Alternatively, class priors can be specified in a ``dict``, e.g. ``{'class1': 0.1, 'class2': 0.3, 'class3': 0.6}``.

        return_scores: bool
            Whether to return the scores of each model on the observation sequence(s).

        Returns
        -------
        prediction(s): str or List[str]
            The predicted label(s) for the observation sequence(s).

            If ``return_scores`` is true, then for each observation sequence, a tuple `(label, scores)` is returned for each label,
            consisting of the `scores` of each HMM and the `label` of the HMM with the best score.
        """
        X = self._val.observation_sequences(X, allow_single=True)
        if isinstance(prior, dict):
            assert len(prior) == len(self._models), 'There must be a class prior for each HMM or class'
            assert all(model.label in prior for model in self._models), 'There must be a class prior for each HMM or class'
            assert all(isinstance(p, (int, float)) for p in prior.values()), 'Class priors must be numerical'
            assert all(0. <= p <= 1. for p in prior.values()), 'Class priors must each be between zero and one'
            assert np.isclose(sum(prior.values()), 1.), 'Class priors must form a probability distribution by summing to one'
        else:
            self._val.one_of(prior, ['frequency', 'uniform'], desc='prior')
        self._val.boolean(return_scores, desc='return_scores')

        try:
            self._models
        except AttributeError as e:
            raise AttributeError('The classifier needs to be fitted before predictions are made') from e

        if prior == 'frequency':
            total_seqs = sum(model.n_seqs for model in self._models)
            prior = {model.label:(model.n_seqs / total_seqs) for model in self._models}
        elif prior == 'uniform':
            prior = {model.label:(1. / len(self._models)) for model in self._models}

        def _map(sequence):
            scores = [(model.label, model.forward(sequence) - np.log(prior[model.label])) for model in self._models]
            best = min(scores, key=lambda x: x[1])
            return (best[0], scores) if return_scores else best[0]

        # Return MAP predictions (and scores) for observation sequence(s)
        return _map(X) if isinstance(X, np.ndarray) else [_map(sequence) for sequence in X]

    def evaluate(self, X, y, prior='frequency', labels=None):
        """Evaluates the performance of the classifier on a batch of observation sequences and their labels.

        Parameters
        ----------
        X: List[numpy.ndarray]
            A list of multiple observation sequences.

        y: List[str]
            A list of labels for the observation sequences.

        prior: {'frequency', 'uniform'} or Dict[str, float]
            How the prior for each model is calculated to perform MAP estimation by scoring with
            the joint probability (or un-normalized posterior) :math:`\mathbb{P}(O, \lambda_c)=\mathbb{P}(O|\lambda_c)\mathbb{P}(\lambda_c)`,
            where the likelihood :math:`\mathbb{P}(O|\lambda_c)` is generated from the models' :func:`~HMM.forward` function.

            - `'frequency'`: Calculate the prior :math:`\mathbb{P}(\lambda_c)` as the proportion of training examples in class :math:`c`.
            - `'uniform'`: Set the priors uniformly such that :math:`\mathbb{P}(\lambda_c)=\\frac{1}{C}` for each class :math:`c\in\{1,\ldots,C\}`.

            Alternatively, class priors can be specified in a ``dict``, e.g. ``{'class1': 0.1, 'class2': 0.3, 'class3': 0.6}``.

        labels: List[str]
            A list of labels for ordering the axes of the confusion matrix.

        Returns
        -------
        accuracy: float
            The categorical accuracy of the classifier on the observation sequences.

        confusion: numpy.ndarray
            The confusion matrix representing the discrepancy between predicted and actual labels.
        """
        X, y = self._val.observation_sequences_and_labels(X, y)

        if labels is not None:
            self._val.list_of_strings(labels, desc='confusion matrix labels')

        # Classify each observation sequence and calculate confusion matrix
        predictions = self.predict(X, prior, return_scores=False)
        cm = confusion_matrix(y, predictions, labels=labels)

        return np.sum(np.diag(cm)) / np.sum(cm), cm

    def as_dict(self):
        """Serializes the :class:`HMMClassifier` object into a `dict`, ready to be stored in JSON format.

        .. note::
            Serializing a :class:`HMMClassifier` implicitly serializes the internal :class:`HMM` or :class:`GMMHMM` objects
            by calling :meth:`HMM.as_dict` or :meth:`GMMHMM.as_dict` and storing all of the model data in a single `dict`.

        Returns
        -------
        serialized: dict
            JSON-ready serialization of the :class:`HMMClassifier` object.

        See Also
        --------
        HMM.as_dict: The serialization function used for individual :class:`HMM` objects.
        GMMHMM.as_dict: The serialization function used for individual :class:`GMMHMM` objects.
        """

        try:
            self._models
        except AttributeError as e:
            raise AttributeError('The classifier needs to be fitted before it can be exported to a dict') from e

        return {'models': [model.as_dict() for model in self._models]}

    def save(self, path):
        """Converts the :class:`HMMClassifier` object into a `dict` and stores it in a JSON file.

        Parameters
        ----------
        path: str
            File path (with or without `.json` extension) to store the JSON-serialized :class:`HMMClassifier` object.

        See Also
        --------
        as_dict: Generates the `dict` that is stored in the JSON file.
        """

        data = self.as_dict()
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)

    @classmethod
    def load(cls, path, random_state=None):
        """Deserializes either a `dict` or JSON serialized :class:`HMMClassifier` object.

        Parameters
        ----------
        path: str
            File path of the serialized JSON data generated by the :meth:`save` method.

        random_state: numpy.random.RandomState, int, optional
            A random state object or seed for reproducible randomness.

        Returns
        -------
        deserialized: :class:`HMMClassifier`
            The deserialized HMM classifier object.

        See Also
        --------
        save: Serializes a :class:`HMMClassifier` into a JSON file.
        as_dict: Generates a `dict` representation of the :class:`HMMClassifier`.
        """

        # Load the serialized HMM classifier as JSON
        with open(path, 'r') as f:
            data = json.load(f)

        clf = cls()
        clf._models = []

        for model in data['models']:
            # Retrieve the type of HMM
            if model['type'] == 'HMM':
                hmm = HMM
            elif model['type'] == 'GMMHMM':
                hmm = GMMHMM
            else:
                raise ValueError("Expected 'type' field to be either 'HMM' or 'GMMHMM'")

            # Deserialize the HMM and add it to the classifier
            clf._models.append(hmm.load(model, random_state=random_state))

        return clf