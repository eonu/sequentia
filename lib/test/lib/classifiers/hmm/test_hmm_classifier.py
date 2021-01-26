import os, pickle, pytest, warnings, os, numpy as np, hmmlearn.hmm
from copy import deepcopy
from sequentia.classifiers import GMMHMM, HMMClassifier, _ErgodicTopology
from ....support import assert_equal, assert_not_equal

# Set seed for reproducible randomness
seed = 0
np.random.seed(seed)
rng = np.random.RandomState(seed)

# Set of possible labels
labels = ['c{}'.format(i) for i in range(5)]

# Create and fit some sample HMMs
hmms = []
for i, label in enumerate(labels):
    hmm = GMMHMM(label=label, n_states=(i + 3), random_state=rng)
    hmm.set_random_initial()
    hmm.set_random_transitions()
    hmm.fit([np.arange((i + j * 20) * 30).reshape(-1, 3) for j in range(2, 5)])
    hmms.append(hmm)

# Create some sample test data and labels
X = [np.arange((i + 2 * 20) * 30).reshape(-1, 3) for i in range(2, 5)]
Y = ['c0', 'c1', 'c1']
x, y = X[0], 'c1'

# Fit a classifier
hmm_clf = HMMClassifier()
hmm_clf.fit(hmms)

# =================== #
# HMMClassifier.fit() #
# =================== #

def test_fit_list():
    """Fit the classifier using a list of HMMs"""
    clf = HMMClassifier()
    clf.fit(hmms)
    assert clf.models_ == hmms

def test_fit_list_empty():
    """Fit the classifier using an empty list"""
    clf = HMMClassifier()
    with pytest.raises(RuntimeError) as e:
        clf.fit([])
    assert str(e.value) == 'The classifier must be fitted with at least one HMM'

def test_fit_list_invalid():
    """Fit the classifier using a list of invalid types"""
    clf = HMMClassifier()
    with pytest.raises(TypeError) as e:
        clf.fit(['a', 'b'])
    assert str(e.value) == 'Expected all models to be GMMHMM objects'

# ======================= #
# HMMClassifier.predict() #
# ======================= #

def test_predict_single_frequency_prior():
    """Predict a single observation sequence with a frequency prior"""
    prediction = hmm_clf.predict(x, prior='frequency', return_scores=False, original_labels=False)
    assert prediction == 0

def test_predict_single_uniform_prior():
    """Predict a single observation sequence with a uniform prior"""
    prediction = hmm_clf.predict(x, prior='uniform', return_scores=False, original_labels=False)
    assert prediction == 0

def test_predict_single_custom_prior():
    """Predict a single observation sequence with a custom prior"""
    prediction = hmm_clf.predict(x, prior=([1e-50]*4+[1-4e-50]), return_scores=False, original_labels=False)
    assert prediction == 4

def test_predict_single_return_scores():
    """Predict a single observation sequence and return the transformed label, with the un-normalized posterior scores"""
    prediction = hmm_clf.predict(x, prior='frequency', return_scores=True, original_labels=False)
    assert isinstance(prediction, tuple)
    assert prediction[0] == 0
    assert_equal(prediction[1], np.array([
        -1225.88304108, -1266.85875999, -1266.96016441, -1226.97939403, -1274.89102844
    ]))

def test_predict_single_original_labels():
    """Predict a single observation sequence and return the original label, without the un-normalized posterior scores"""
    prediction = hmm_clf.predict(x, prior='uniform', return_scores=False, original_labels=True)
    assert prediction == 'c0'

def test_predict_single_return_scores_original_labels():
    """Predict a single observation sequence and return the original label, with the un-normalized posterior scores"""
    prediction = hmm_clf.predict(x, prior='frequency', return_scores=True, original_labels=True)
    assert isinstance(prediction, tuple)
    assert prediction[0] == 'c0'
    assert_equal(prediction[1], np.array([
        -1225.88304108, -1266.85875999, -1266.96016441, -1226.97939403, -1274.89102844
    ]))

def test_predict_multiple_frequency_prior():
    """Predict multiple observation sequences with a frequency prior"""
    predictions = hmm_clf.predict(X, prior='frequency', return_scores=False, original_labels=False)
    assert_equal(predictions, np.array([0, 0, 0]))

def test_predict_multiple_uniform_prior():
    """Predict multiple observation sequences with a uniform prior"""
    predictions = hmm_clf.predict(X, prior='uniform', return_scores=False, original_labels=False)
    assert_equal(predictions, np.array([0, 0, 0]))

def test_predict_multiple_custom_prior():
    """Predict multiple observation sequences with a custom prior"""
    predictions = hmm_clf.predict(X, prior=([1-4e-50]+[1e-50]*4), return_scores=False, original_labels=False)
    assert_equal(predictions, np.array([0, 0, 0]))

def test_predict_multiple_return_scores():
    """Predict multiple observation sequences and return the transformed labels, with the un-normalized posterior scores"""
    predictions = hmm_clf.predict(X, prior='frequency', return_scores=True, original_labels=False)
    assert isinstance(predictions, tuple)
    assert_equal(predictions[0], np.array([0, 0, 0]))
    assert_equal(predictions[1], np.array([
        [-1225.88304108, -1266.85875999, -1266.96016441, -1226.97939403, -1274.89102844],
        [-1254.2158035 , -1299.37586652, -1299.75108935, -1255.8359274 , -1308.71071239],
        [-1282.57116414, -1330.90436081, -1331.63379359, -1284.79130597, -1342.45717804]
    ]))

def test_predict_multiple_original_labels():
    """Predict multiple observation sequences and return the original labels, without the un-normalized posterior scores"""
    predictions = hmm_clf.predict(X, prior='frequency', return_scores=False, original_labels=True)
    assert all(np.equal(predictions.astype(object), np.array(['c0', 'c0', 'c0'], dtype=object)))

def test_predict_multiple_return_scores_original_labels():
    """Predict multiple observation sequences and return the original labels, with the un-normalized posterior scores"""
    predictions = hmm_clf.predict(X, prior='frequency', return_scores=True, original_labels=True)
    assert isinstance(predictions, tuple)
    assert all(np.equal(predictions[0].astype(object), np.array(['c0', 'c0', 'c0'], dtype=object)))
    assert_equal(predictions[1], np.array([
        [-1225.88304108, -1266.85875999, -1266.96016441, -1226.97939403, -1274.89102844],
        [-1254.2158035 , -1299.37586652, -1299.75108935, -1255.8359274 , -1308.71071239],
        [-1282.57116414, -1330.90436081, -1331.63379359, -1284.79130597, -1342.45717804]
    ]))

# ======================== #
# HMMClassifier.evaluate() #
# ======================== #

def test_evaluate():
    """Evaluate performance on some observation sequences and labels"""
    acc, cm = hmm_clf.evaluate(X, Y, prior='frequency')
    assert acc == 1 / 3
    assert_equal(cm, np.array([
        [1, 0, 0, 0, 0],
        [2, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ]))

# ==================== #
# HMMClassifier.save() #
# ==================== #

def test_save_unfitted():
    """Save an unfitted HMMClassifier object."""
    try:
        with pytest.raises(AttributeError) as e:
            HMMClassifier().save('test.pkl')
        assert str(e.value) == 'No models available - the classifier must be fitted first'
    finally:
        if os.path.exists('test.pkl'):
            os.remove('test.pkl')

def test_save_fitted():
    """Save a fitted HMMClassifier object."""
    try:
        hmm_clf.save('test.pkl')
        assert os.path.isfile('test.pkl')
    finally:
        os.remove('test.pkl')

# ==================== #
# HMMClassifier.load() #
# ==================== #

def test_load_invalid_format():
    """Load a HMMClassifier from an illegally formatted file"""
    try:
        with open('test', 'w') as f:
            f.write('illegal')
        with pytest.raises(pickle.UnpicklingError) as e:
            HMMClassifier.load('test')
    finally:
        os.remove('test')

def test_load_valid():
    """Load a serialized HMMClassifier"""
    try:
        hmm_clf.save('test.pkl')
        clf = HMMClassifier.load('test.pkl')
        # Check that all fields are still the same
        assert isinstance(clf, HMMClassifier)
        assert all(isinstance(model, GMMHMM) for model in clf.models_)
        assert [model.label for model in clf.models_] == labels
        assert list(clf.encoder_.classes_) == labels
        predictions = clf.predict(X, prior='frequency', return_scores=True, original_labels=True)
        assert isinstance(predictions, tuple)
        assert all(np.equal(predictions[0].astype(object), np.array(['c0', 'c0', 'c0'], dtype=object)))
        assert_equal(predictions[1], np.array([
            [-1225.88304108, -1266.85875999, -1266.96016441, -1226.97939403, -1274.89102844],
            [-1254.2158035 , -1299.37586652, -1299.75108935, -1255.8359274 , -1308.71071239],
            [-1282.57116414, -1330.90436081, -1331.63379359, -1284.79130597, -1342.45717804]
        ]))
    finally:
        os.remove('test.pkl')