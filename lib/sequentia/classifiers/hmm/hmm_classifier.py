import tqdm, tqdm.auto, numpy as np, pickle
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from .gmmhmm import GMMHMM
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from ...internals import _Validator

class HMMClassifier:
    """A classifier that combines individual :class:`~GMMHMM` objects, which each model isolated sequences from a different class.

    Attributes
    ----------
    models: list of GMMHMM
        A collection of the :class:`~GMMHMM` objects to use for classification.

    encoder: sklearn.preprocessing.LabelEncoder
        The label encoder fitted on the set of ``classes`` provided during instantiation.

    classes: numpy.ndarray (str/numeric)
        The complete set of possible classes/labels.
    """

    def __init__(self):
        self._val = _Validator()

    def fit(self, models):
        """Fits the classifier with a collection of :class:`~GMMHMM` objects.

        Parameters
        ----------
        models: array-like of GMMHMM
            A collection of :class:`~GMMHMM` objects to use for classification.
        """

        models = list(self._val.iterable(models, 'models'))
        if not all(isinstance(model, GMMHMM) for model in models):
            raise TypeError('Expected all models to be GMMHMM objects')

        if len(models) > 0:
            self._models = models
        else:
            raise RuntimeError('The classifier must be fitted with at least one HMM')

        self._encoder = LabelEncoder()
        self._encoder.fit([model.label for model in models])

    def predict(self, X, prior='frequency', return_scores=False, original_labels=True, verbose=True, n_jobs=1):
        """Predicts the label for an observation sequence (or multiple sequences) according to maximum likelihood or posterior scores.

        Parameters
        ----------
        X: numpy.ndarray (float) or list of numpy.ndarray (float)
            An individual observation sequence or a list of multiple observation sequences.

        prior: {'frequency', 'uniform'} or array-like of float
            How the prior probability for each model is calculated to perform MAP estimation by scoring with
            the joint probability (or un-normalized posterior) :math:`\\mathbb{P}(O, \\lambda_c)=\\mathbb{P}(O|\\lambda_c)\\mathbb{P}(\\lambda_c)`.

            - `'frequency'`: Calculate the prior probability :math:`\\mathbb{P}(\\lambda_c)` as the proportion of training examples in class :math:`c`.
            - `'uniform'`: Set the priors uniformly such that :math:`\\mathbb{P}(\\lambda_c)=\\frac{1}{C}` for each class :math:`c\\in\\{1,\\ldots,C\\}` (**equivalent to ignoring the prior**).

            Alternatively, class prior probabilities can be specified in an iterable of floats, e.g. `[0.1, 0.3, 0.6]`.

        return_scores: bool
            Whether to return the scores of each model on the observation sequence(s).

        original_labels: bool
            Whether to inverse-transform the labels to their original encoding.

        verbose: bool
            Whether to display a progress bar or not.

            .. note::
                If both ``verbose=True`` and ``n_jobs > 1``, then the progress bars for each process
                are always displayed in the console, regardless of where you are running this function from
                (e.g. a Jupyter notebook).

        n_jobs: int > 0 or -1
            | The number of jobs to run in parallel.
            | Setting this to -1 will use all available CPU cores.

        Returns
        -------
        prediction(s): str/numeric or :class:`numpy:numpy.ndarray` (str/numeric)
            The predicted label(s) for the observation sequence(s).

            If ``original_labels`` is true, then the returned labels are inverse-transformed into their original encoding.

        scores: :class:`numpy:numpy.ndarray` (float)
            An :math:`N\\times M` matrix of scores (log un-normalized posteriors), for each of the :math:`N` observation sequences,
            for each of the :math:`M` HMMs. Only returned if ``return_scores`` is true.
        """
        try:
            self._models
        except AttributeError as e:
            raise AttributeError('The classifier needs to be fitted before predictions are made') from e

        X = self._val.observation_sequences(X, allow_single=True)
        if not isinstance(prior, str):
            self._val.iterable(prior, 'prior')
            assert len(prior) == len(self._models), 'There must be a class prior for each HMM or class'
            assert all(isinstance(p, (int, float)) for p in prior), 'Class priors must be numerical'
            assert all(0. <= p <= 1. for p in prior), 'Class priors must each be between zero and one'
            assert np.isclose(sum(prior), 1.), 'Class priors must form a probability distribution by summing to one'
        else:
            self._val.one_of(prior, ['frequency', 'uniform'], desc='prior')
        self._val.boolean(return_scores, desc='return_scores')
        self._val.boolean(original_labels, desc='original_labels')
        self._val.boolean(verbose, desc='verbose')
        self._val.restricted_integer(n_jobs, lambda x: x == -1 or x > 0, 'number of jobs', '-1 or greater than zero')

        # Create look-up for prior probabilities
        if prior == 'frequency':
            total_seqs = sum(model.n_seqs for model in self._models)
            prior = {model.label:(model.n_seqs / total_seqs) for model in self._models}
        elif prior == 'uniform':
            prior = {model.label:(1. / len(self._models)) for model in self._models}
        else:
            prior = {model.label:prior[self._encoder.transform([model.label]).item()] for model in self._models}

        # Convert single observation sequence to a singleton list
        X = [X] if isinstance(X, np.ndarray) else X

        # Lambda for calculating the log un-normalized posteriors as a sum of the log forward probabilities (likelihoods) and log priors
        posteriors = lambda x: np.array([model.forward(x) + np.log(prior[model.label]) for model in self._models])

        # Calculate log un-normalized posteriors as a sum of the log forward probabilities (likelihoods) and log priors
        # Perform the MAP classification rule and return labels to original encoding if necessary
        n_jobs = min(cpu_count() if n_jobs == -1 else n_jobs, len(X))
        X_chunks = [list(chunk) for chunk in np.array_split(np.array(X, dtype=object), n_jobs)]
        scores = Parallel(n_jobs=n_jobs)(delayed(self._chunk_predict)(i+1, posteriors, chunk, verbose) for i, chunk in enumerate(X_chunks))
        scores = np.concatenate(scores)
        best_idxs = np.atleast_1d(scores.argmax(axis=1))
        labels = self._encoder.inverse_transform(best_idxs) if original_labels else best_idxs

        if len(X) == 1:
            return (labels.item(), scores.flatten()) if return_scores else labels.item()
        else:
            return (labels, scores) if return_scores else labels

    def evaluate(self, X, y, prior='frequency', verbose=True, n_jobs=1):
        """Evaluates the performance of the classifier on a batch of observation sequences and their labels.

        Parameters
        ----------
        X: list of numpy.ndarray (float)
            A list of multiple observation sequences.

        y: array-like of str/numeric
            An iterable of labels for the observation sequences.

        prior: {'frequency', 'uniform'} or array-like of float
            How the prior probability for each model is calculated to perform MAP estimation by scoring with
            the joint probability (or un-normalized posterior) :math:`\\mathbb{P}(O, \\lambda_c)=\\mathbb{P}(O|\\lambda_c)\\mathbb{P}(\\lambda_c)`.

            - `'frequency'`: Calculate the prior probability :math:`\\mathbb{P}(\\lambda_c)` as the proportion of training examples in class :math:`c`.
            - `'uniform'`: Set the priors uniformly such that :math:`\\mathbb{P}(\\lambda_c)=\\frac{1}{C}` for each class :math:`c\\in\\{1,\\ldots,C\\}` (**equivalent to ignoring the prior**).

            Alternatively, class prior probabilities can be specified in an iterable of floats, e.g. `[0.1, 0.3, 0.6]`.

        verbose: bool
            Whether to display a progress bar or not.

            .. note::
                If both ``verbose=True`` and ``n_jobs > 1``, then the progress bars for each process
                are always displayed in the console, regardless of where you are running this function from
                (e.g. a Jupyter notebook).

        n_jobs: int > 0 or -1
            | The number of jobs to run in parallel.
            | Setting this to -1 will use all available CPU cores.

        Returns
        -------
        accuracy: float
            The categorical accuracy of the classifier on the observation sequences.

        confusion: :class:`numpy:numpy.ndarray` (int)
            The confusion matrix representing the discrepancy between predicted and actual labels.
        """
        X, y = self._val.observation_sequences_and_labels(X, y)
        predictions = self.predict(X, prior=prior, return_scores=False, original_labels=False, verbose=verbose, n_jobs=n_jobs)
        cm = confusion_matrix(self._encoder.transform(y), predictions, labels=self._encoder.transform(self._encoder.classes_))
        return np.sum(np.diag(cm)) / np.sum(cm), cm

    def save(self, path):
        """Serializes the :class:`HMMClassifier` object by pickling.

        Parameters
        ----------
        path: str
            File path (usually with `.pkl` extension) to store the serialized :class:`HMMClassifier` object.
        """
        try:
            self._models
        except AttributeError:
            raise RuntimeError('The classifier needs to be fitted before it can be saved')

        with open(path, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path):
        """Deserializes a :class:`HMMClassifier` object which was serialized with the :meth:`save` function.

        Parameters
        ----------
        path: str
            File path of the serialized data generated by the :meth:`save` method.

        Returns
        -------
        deserialized: :class:`HMMClassifier`
            The deserialized HMM classifier object.
        """
        with open(path, 'rb') as file:
            return pickle.load(file)

    def _chunk_predict(self, process, posteriors, chunk, verbose): # Requires fit
        """Makes predictions (scores) for a chunk of the observation sequences, for a given subprocess."""
        return np.array([posteriors(x) for x in tqdm.auto.tqdm(
            chunk, desc='Classifying examples (process {})'.format(process),
            disable=not(verbose), position=process-1
        )])

    @property
    def models(self):
        try:
            return self._models
        except AttributeError as e:
            raise AttributeError('No models available - the classifier must be fitted first') from e

    @property
    def encoder(self):
        try:
            return self._encoder
        except AttributeError as e:
            raise AttributeError('No label encoder has been defined - the classifier must be fitted first') from e

    @property
    def classes(self):
        return self.encoder.classes_

    def __repr__(self):
        module = self.__class__.__module__
        out = '{}{}('.format('' if module == '__main__' else '{}.'.format(module), self.__class__.__name__)
        try:
            self._models
            out += 'models=[\n  '
            out += ',\n  '.join(repr(model) for model in self._models)
            out += '\n]'
        except AttributeError:
            pass
        return out + ')'