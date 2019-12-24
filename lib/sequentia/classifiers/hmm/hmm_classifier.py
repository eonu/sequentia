import numpy as np
from .hmm import HMM
from sklearn.metrics import confusion_matrix
from typing import Dict, Union, List, Tuple, Any
from ...internals import Validator

class HMMClassifier:
    """An ensemble classifier that combines individual HMMs which model isolated sequences from different classes.

    Example:
        >>> import numpy as np
        >>> from sequentia.classifiers import HMM, HMMClassifier
        >>> ​
        >>> # Create and fit some sample HMMs
        >>> hmms = []
        >>> for i in range(5):
        >>>     hmm = HMM(label=f'class{i}', n_states=(i + 3), topology='left-right')
        >>>     hmm.set_random_initial()
        >>>     hmm.set_random_transitions()
        >>>     hmm.fit([np.arange((i + j * 20) * 30).reshape(-1, 3) for j in range(1, 4)])
        >>>     hmms.append(hmm)
        >>> ​
        >>> # Create some sample test data and labels
        >>> X = [np.random.random((10 * i, 3)) for i in range(1, 4)]
        >>> y = ['class0', 'class1', 'class1']
        >>> ​
        >>> # Create a classifier and calculate predictions and evaluations
        >>> clf = HMMClassifier()
        >>> clf.fit(hmms)
        >>> predictions = clf.predict(X)
        >>> f1, confusion = clf.evaluate(X, y)
    """

    def __init__(self):
        self._val = Validator()

    def fit(self, models: Union[List[HMM], Dict[Any, HMM]]):
        """
        Parameters:
            models {list(HMM),dict(HMM)} - A collection of HMM objects to use for classification.
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

    def predict(self, X: Union[np.ndarray, List[np.ndarray]], prior=True, return_scores=False) -> Union[str, List[str]]:
        """Predicts the label for an observation sequence (or multiple sequences) according to maximum likelihood or posterior scores.

        Parameters:
            X {numpy.ndarray, list(numpy.ndarray)} - An individual observation sequence or
                a list of multiple observation sequences.
            prior {bool} - Whether to calculate a prior and perform MAP estimation. If this parameter is set
                to False, then the negative log likelihoods generated from the models' `forward` function are used.
            return_scores {bool} - Whether to return the scores of each model on the observation sequence(s).

        Returns {str, list(str)}:
            The predicted labels for the observation sequence(s).
        """
        self._val.boolean(prior, desc='prior')
        self._val.boolean(return_scores, desc='return_scores')
        self._val.observation_sequences(X, allow_single=True)

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

    def evaluate(self, X: List[np.ndarray], y: List[str], prior=True, labels=None) -> Tuple[float, np.ndarray]:
        """Evaluates the performance of the classifier on a batch of observation sequences and their labels.

        Parameters:
            X {list(numpy.ndarray)} - A list of multiple observation sequences.
            y {list(str)} - A list of labels for the observation sequences.
            prior {bool} - Whether to calculate a prior and perform MAP estimation. If this parameter is set
                to False, then the negative log likelihoods generated from the models' `forward` function are used.
            labels {list(str)} - A list of labels for ordering the axes of the confusion matrix.

        Return: {tuple(float, numpy.ndarray)}
            - The categorical accuracy of the classifier on the observation sequences.
            - A confusion matrix representing the discrepancy between predicted and actual labels.
        """
        self._val.observation_sequences_and_labels(X, y)
        self._val.boolean(prior, desc='prior')

        if labels is not None:
            self._val.list_of_strings(labels, desc='confusion matrix labels')

        # Classify each observation sequence and calculate confusion matrix
        predictions = self.predict(X, prior, return_scores=False)
        cm = confusion_matrix(y, predictions, labels=labels)

        return np.sum(np.diag(cm)) / np.sum(cm), cm