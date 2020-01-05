import pytest
import warnings
import numpy as np
from copy import deepcopy
from sequentia.classifiers import HMM, HMMClassifier
from ....support import assert_equal, assert_not_equal

# Set seed for reproducible randomness
seed = 0
np.random.seed(seed)
rng = np.random.RandomState(seed)

# Set of possible labels
labels = ['c{}'.format(i) for i in range(5)]

# Create and fit some sample HMMs
hmm_list, hmm_dict = [], {}
for i, label in enumerate(labels):
    hmm = HMM(label=label, n_states=(i + 3), random_state=rng)
    hmm.set_random_initial()
    hmm.set_random_transitions()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        hmm.fit([np.arange((i + j * 20) * 30).reshape(-1, 3) for j in range(2, 5)])
    hmm_list.append(hmm)
    hmm_dict[label] = hmm

# Create some sample test data and labels
X = [np.arange((i + 2 * 20) * 30).reshape(-1, 3) for i in range(2, 5)]
Y = ['c0', 'c1', 'c1']
x, y = X[0], 'c1'

# Fit a classifier
hmm_clf = HMMClassifier()
hmm_clf.fit(hmm_list)

# =================== #
# HMMClassifier.fit() #
# =================== #

def test_fit_list():
    """Fit the classifier using a list of HMMs"""
    clf = HMMClassifier()
    clf.fit(hmm_list)
    assert clf._models == hmm_list

def test_fit_list_empty():
    """Fit the classifier using an empty list"""
    clf = HMMClassifier()
    with pytest.raises(RuntimeError) as e:
        clf.fit([])
    assert str(e.value) == 'Must fit the classifier with at least one HMM'

def test_fit_list_invalid():
    """Fit the classifier using a list of invalid types"""
    clf = HMMClassifier()
    with pytest.raises(TypeError) as e:
        clf.fit(['a', 'b'])
    assert str(e.value) == 'Expected all models to be HMM objects'

def test_fit_dict():
    """Fit the classifier using a dict of HMMs"""
    clf = HMMClassifier()
    clf.fit(hmm_dict)
    assert clf._models == list(hmm_dict.values())

def test_fit_dict_empty():
    """Fit the classifier using an empty dict"""
    clf = HMMClassifier()
    with pytest.raises(RuntimeError) as e:
        clf.fit({})
    assert str(e.value) == 'Must fit the classifier with at least one HMM'

def test_fit_dict_invalid():
    """Fit the classifier using a dict of invalid types"""
    clf = HMMClassifier()
    with pytest.raises(TypeError) as e:
        clf.fit({'a': 0, 'b': 0})
    assert str(e.value) == 'Expected all models to be HMM objects'

def test_fit_invalid():
    """Fit the classifier with an invalid"""
    clf = HMMClassifier()
    with pytest.raises(TypeError) as e:
        clf.fit(tuple(hmm_list))
    assert str(e.value) == 'Expected models to be a list or dict of HMM objects'

# ======================= #
# HMMClassifier.predict() #
# ======================= #

def test_predict_single_with_prior_with_return_scores():
    """Predict a single observation sequence with a prior and returned scores"""
    prediction = hmm_clf.predict(x, prior=True, return_scores=True)
    assert isinstance(prediction, tuple)
    assert isinstance(prediction[0], str)
    assert isinstance(prediction[1], list)
    assert isinstance(prediction[1][0], tuple)
    assert isinstance(prediction[1][0][0], str)
    assert isinstance(prediction[1][0][1], float)
    assert len(prediction[1]) == 5

def test_predict_single_with_prior_no_return_scores():
    """Predict a single observation sequence with a prior and no returned scores"""
    prediction = hmm_clf.predict(x, prior=True, return_scores=False)
    assert isinstance(prediction, str)

def test_predict_single_no_prior_with_return_scores():
    """Predict a single observation sequence with no prior and returned scores"""
    prediction = hmm_clf.predict(x, prior=False, return_scores=True)
    assert isinstance(prediction, tuple)
    assert isinstance(prediction[0], str)
    assert isinstance(prediction[1], list)
    assert isinstance(prediction[1][0], tuple)
    assert isinstance(prediction[1][0][0], str)
    assert isinstance(prediction[1][0][1], float)
    assert len(prediction[1]) == 5

def test_predict_single_no_prior_no_return_scores():
    """Predict a single observation sequence with no prior and no returned scores"""
    prediction = hmm_clf.predict(x, prior=False, return_scores=False)
    assert isinstance(prediction, str)

def test_predict_multiple_with_prior_with_return_scores():
    """Predict multiple observation sequences with a prior and returned scores"""
    predictions = hmm_clf.predict(X, prior=True, return_scores=True)
    assert isinstance(predictions, list)
    assert len(predictions) == 3
    # First prediction
    assert isinstance(predictions[0], tuple)
    assert isinstance(predictions[0][0], str)
    assert isinstance(predictions[0][1], list)
    assert len(predictions[0][1]) == 5
    # First score
    assert isinstance(predictions[0][1][0], tuple)
    assert isinstance(predictions[0][1][0][0], str)
    assert isinstance(predictions[0][1][0][1], float)

def test_predict_multiple_with_prior_no_return_scores():
    """Predict multiple observation sequences with a prior and no returned scores"""
    predictions = hmm_clf.predict(X, prior=True, return_scores=False)
    assert isinstance(predictions, list)
    assert len(predictions) == 3

def test_predict_multiple_no_prior_with_return_scores():
    """Predict multiple observation sequences with no prior and returned scores"""
    predictions = hmm_clf.predict(X, prior=False, return_scores=True)
    assert isinstance(predictions, list)
    assert len(predictions) == 3
    # First prediction
    assert isinstance(predictions[0], tuple)
    assert isinstance(predictions[0][0], str)
    assert isinstance(predictions[0][1], list)
    assert len(predictions[0][1]) == 5
    # First score
    assert isinstance(predictions[0][1][0], tuple)
    assert isinstance(predictions[0][1][0][0], str)
    assert isinstance(predictions[0][1][0][1], float)

def test_predict_single_no_prior_no_return_scores():
    """Predict multiple observation sequences with no prior and no returned scores"""
    predictions = hmm_clf.predict(X, prior=False, return_scores=False)
    assert isinstance(predictions, list)
    assert len(predictions) == 3

# ======================== #
# HMMClassifier.evaluate() #
# ======================== #

def test_evaluate_with_prior_with_labels():
    """Evaluate with a prior and confusion matrix labels"""
    acc, cm = hmm_clf.evaluate(X, Y, prior=True, labels=['c0', 'c1', 'c2', 'c3', 'c4'])
    assert isinstance(acc, float)
    assert isinstance(cm, np.ndarray)
    assert cm.shape == (5, 5)

def test_evaluate_with_prior_no_labels():
    """Evaluate with a prior and no confusion matrix labels"""
    acc, cm = hmm_clf.evaluate(X, Y, prior=True, labels=None)
    assert isinstance(acc, float)
    assert isinstance(cm, np.ndarray)

def test_evaluate_no_prior_with_labels():
    """Evaluate with no prior and confusion matrix labels"""
    acc, cm = hmm_clf.evaluate(X, Y, prior=False, labels=['c0', 'c1', 'c2', 'c3', 'c4'])
    assert isinstance(acc, float)
    assert isinstance(cm, np.ndarray)
    assert cm.shape == (5, 5)

def test_evaluate_no_prior_no_labels():
    """Evaluate with no prior and no confusion matrix labels"""
    acc, cm = hmm_clf.evaluate(X, Y, prior=False, labels=None)
    assert isinstance(acc, float)
    assert isinstance(cm, np.ndarray)