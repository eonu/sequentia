import numpy as np
from .hmm import HMM
from sklearn.metrics import confusion_matrix
from ...internals import _Validator

class HMMClassifier:
    """A classifier that combines individual :class:`~HMM` objects, which model isolated sequences from different classes."""

    def __init__(self):
        self._val = _Validator()

    def fit(self, models):
        """Fits the classifier with a collection of :class:`~HMM` objects.

        Parameters
        ----------
        models: List[HMM] or Dict[Any, HMM]
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

    def predict(self, X, prior=True, return_scores=False):
        """Predicts the label for an observation sequence (or multiple sequences) according to maximum likelihood or posterior scores.

        Parameters
        ----------
        X: numpy.ndarray or List[numpy.ndarray]
            An individual observation sequence or a list of multiple observation sequences.

        prior: bool
            Whether to calculate a prior for each model and perform MAP estimation by scoring with
            the joint probability (or un-normalized posterior) :math:`\mathbb{P}(O, \lambda_c)=\mathbb{P}(O|\lambda_c)\mathbb{P}(\lambda_c)`.

            If this parameter is set to false, then the negative log likelihoods
            :math:`\mathbb{P}(O|\lambda_c)` generated from the models' :func:`~HMM.forward` function are used.

        return_scores: bool
            Whether to return the scores of each model on the observation sequence(s).

        Returns
        -------
        prediction(s): str or List[str]
            The predicted label(s) for the observation sequence(s).

            If ``return_scores`` is true, then for each observation sequence, a tuple `(label, scores)` is returned for each label,
            consisting of the `scores` of each HMM and the `label` of the HMM with the best score.
        """
        self._val.boolean(prior, desc='prior')
        self._val.boolean(return_scores, desc='return_scores')
        X = self._val.observation_sequences(X, allow_single=True)

        try:
            self._models
        except AttributeError as e:
            raise AttributeError('The classifier needs to be fitted before predictions are made') from e

        total_seqs = sum(model.n_seqs for model in self._models)

        if isinstance(X, np.ndarray): # Single observation sequence
            scores = [(model.label, model.forward(X) - np.log(model.n_seqs / total_seqs) * prior) for model in self._models]
            best = min(scores, key=lambda x: x[1])
            return (best[0], scores) if return_scores else best[0]
        else: # Multiple observation sequences
            predictions = []
            for x in X:
                scores = [(model.label, model.forward(x) - np.log(model.n_seqs / total_seqs) * prior) for model in self._models]
                best = min(scores, key=lambda x: x[1])
                predictions.append((best[0], scores) if return_scores else best[0])
            return predictions

    def evaluate(self, X, y, prior=True, labels=None):
        """Evaluates the performance of the classifier on a batch of observation sequences and their labels.

        Parameters
        ----------
        X: List[numpy.ndarray]
            A list of multiple observation sequences.

        y: List[str]
            A list of labels for the observation sequences.

        prior: bool
            Whether to calculate a prior for each model and perform MAP estimation by scoring with
            the joint probability (or un-normalized posterior) :math:`\mathbb{P}(O, \lambda_c)=\mathbb{P}(O|\lambda_c)\mathbb{P}(\lambda_c)`.

            If this parameter is set to false, then the negative log likelihoods
            :math:`\mathbb{P}(O|\lambda_c)` generated from the models' :func:`~HMM.forward` function are used.

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
        self._val.boolean(prior, desc='prior')

        if labels is not None:
            self._val.list_of_strings(labels, desc='confusion matrix labels')

        # Classify each observation sequence and calculate confusion matrix
        predictions = self.predict(X, prior, return_scores=False)
        cm = confusion_matrix(y, predictions, labels=labels)

        return np.sum(np.diag(cm)) / np.sum(cm), cm