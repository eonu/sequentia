import numpy as np
from .hmm import HMM
from sklearn.metrics import confusion_matrix
from typing import Dict, Union, List, Tuple, Any

class HMMClassifier:
    """An ensemble classifier that combines individual HMMs which model isolated sequences from different classes.

    Examples:
        >>> import numpy as np
        >>> from sequentia.classifiers import HMM, HMMClassifier
        >>>
        >>> # Create some sample HMMs
        >>> hmms = []
        >>> for i in range(5):
        >>>     hmm = HMM(label=f'class{i}', n_states=(i + 3), topology='left-right')
        >>>     hmm.set_random_initial()
        >>>     hmm.set_random_transitions()
        >>>     hmm.fit([np.arange((i + j * 20) * 30).reshape(-1, 3) for j in range(1, 4)])
        >>>     hmms.append(hmm)
        >>>
        >>> # Create some sample test data and labels
        >>> X = [np.random.random((10 * i, 3)) for i in range(1, 4)]
        >>> y = ['class0', 'class1', 'class1']
        >>>
        >>> # Create a classifier and calculate predictions and evaluations
        >>> clf = HMMClassifier()
        >>> clf.fit(hmms)
        >>> predictions = clf.predict(X)
        >>> f1, confusion = clf.evaluate(X, y, metric='f1')
    """

    def fit(self, models: Union[List[HMM], Dict[Any, HMM]]):
        """
        Parameters:
            models {list(HMM),dict(HMM)} - A collection of HMM objects to use for classification.
        """
        if isinstance(models, list):
            if not all(isinstance(model, HMM) for model in models):
                raise TypeError('Expected all models to be HMM objects')
            self._models = models
        elif isinstance(models, dict):
            values = list(models.values())
            if not all(isinstance(model, HMM) for model in values):
                raise TypeError('Expected all models to be HMM objects')
            self._models = values
        else:
            raise TypeError('Expected `models` to be a list or dict of HMM objects')

    def predict(self, X: Union[np.ndarray, List[np.ndarray]], prior=True, return_scores=False) -> Union[np.ndarray, List[np.ndarray]]:
        """Predicts the label for an observation sequence (or multiple sequences) according to maximum likelihood or posterior scores.

        Parameters:
            X {numpy.ndarray, list(numpy.ndarray)} - An individual observation sequence or
                a list of multiple observation sequences.
            prior {bool} - Whether to calculate a prior and perform MAP estimation. If this parameter is set
                to False, then the negative log likelihoods generated from the models' `forward` function are used.
            return_scores {bool} - Whether to return the scores of each model on the observation sequence(s).

        Returns:
            The predicted labels for the observation sequence(s).
        """
        if not isinstance(prior, bool):
            raise TypeError('Expected `prior` to be a bool')

        if not isinstance(return_scores, bool):
            raise TypeError('Expected `return_scores` to be a bool')

        if isinstance(X, np.ndarray):
            if not X.ndim == 2:
                raise ValueError('Sequence of observations must be two-dimensional')

            total_seqs = sum(model.n_seqs for model in self._models)
            scores = [(model.label, model.forward(X) - np.log(model.n_seqs / total_seqs) * prior) for model in self._models]
            best = min(scores, key=lambda x: x[1])
            return (best[0], scores) if return_scores else best[0]
        elif isinstance(X, list):
            if not all(isinstance(sequence, np.ndarray) for sequence in X):
                raise TypeError('Each observation sequence must be a numpy.ndarray')
            if not all(sequence.ndim == 2 for sequence in X):
                raise ValueError('Each observation sequence must be two-dimensional')
            if not all(sequence.shape[1] == X[0].shape[1] for sequence in X):
                raise ValueError('Each observation sequence must have the same dimensionality')

            predictions = []
            total_seqs = sum(model.n_seqs for model in self._models)
            for x in X:
                scores = [(model.label, model.forward(x) - np.log(model.n_seqs / total_seqs) * prior) for model in self._models]
                best = min(scores, key=lambda x: x[1])
                predictions.append((best[0], scores) if return_scores else best[0])
            return predictions
        else:
            raise TypeError('Expected a single observation sequence (numpy.ndarray) or list of observation sequences')

    def evaluate(self, X: List[np.ndarray], y: List[str], prior=True, metric='accuracy', labels=None) -> Tuple[float, np.ndarray]:
        """Evaluates the performance of the classifier on a batch of observation sequences and their labels.

        Parameters:
            X {list(numpy.ndarray)} - A list of multiple observation sequences.
            y {list(str)} - A list of labels for the observation sequences.
            prior {bool} - Whether to calculate a prior and perform MAP estimation. If this parameter is set
                to False, then the negative log likelihoods generated from the models' `forward` function are used.
            metric {str} - A performance metric for the classification - one of
                'accuracy' (categorical accuracy) or 'f1' (F1 score = harmonic mean of precision and recall).
            labels {list(str)} - A list of labels for ordering the axes of the confusion matrix.

        Return: {tuple(float, numpy.ndarray)}
            - The specified performance result: either categorical accuracy or F1 score.
            - A confusion matrix representing the discrepancy between predicted and actual labels.
        """
        if isinstance(X, list):
            if not all(isinstance(sequence, np.ndarray) for sequence in X):
                raise TypeError('Each observation sequence must be a numpy.ndarray')
            if not all(sequence.ndim == 2 for sequence in X):
                raise ValueError('Each observation sequence must be two-dimensional')
            if not all(sequence.shape[1] == X[0].shape[1] for sequence in X):
                raise ValueError('Each observation sequence must have the same dimensionality')
        else:
            raise TypeError('Expected a list of observation sequences, each of type numpy.ndarray')

        if isinstance(y, list):
            if not all(isinstance(label, str) for label in y):
                raise ValueError('Expected all labels to be strings')
        else:
            raise ValueError('Expected labels to be a list of strings')

        if not len(X) == len(y):
            raise ValueError('Expected the same number of observation sequences and labels')

        if not isinstance(prior, bool):
            raise TypeError('Expected `prior` to be a bool')

        if metric not in ['accuracy', 'f1']:
            raise ValueError("Expected `metric` to be one of 'accuracy' or 'f1'")

        if labels is not None:
            if isinstance(labels, list):
                if not all(isinstance(label, str) for label in labels):
                    raise ValueError('Expected all confusion matrix labels to be strings')
            else:
                raise ValueError('Expected confusion matrix labels to be a list of strings')

        # Classify each observation sequence
        predictions = self.predict(X, prior, return_scores=False)

        # Calculate confusion matrix and precision and recall
        cm = confusion_matrix(y, predictions, labels=labels)
        precision = np.mean(np.diag(cm) / np.sum(cm, axis=0))
        recall = np.mean(np.diag(cm) / np.sum(cm, axis=1))

        if metric == 'accuracy':
            return np.sum(np.diag(cm)) / len(predictions), cm
        elif metric == 'f1':
            return 2.0 * precision * recall / (precision + recall), cm