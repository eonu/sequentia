import pytest
import warnings
import numpy as np
from copy import deepcopy
import pomegranate as pg
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
    assert prediction == ('c0', [
        ('c0', -3587.3909579390056),
        ('c1', -3532.9727085623044),
        ('c2', -3532.837286683744),
        ('c3', -3550.3438107711895),
        ('c4', -3551.2566620927946)
    ])

def test_predict_single_with_prior_no_return_scores():
    """Predict a single observation sequence with a prior and no returned scores"""
    prediction = hmm_clf.predict(x, prior=True, return_scores=False)
    assert prediction == 'c0'

def test_predict_single_no_prior_with_return_scores():
    """Predict a single observation sequence with no prior and returned scores"""
    prediction = hmm_clf.predict(x, prior=False, return_scores=True)
    assert prediction == ('c0', [
        ('c0', -3589.00039585144),
        ('c1', -3534.5821464747387),
        ('c2', -3534.4467245961782),
        ('c3', -3551.9532486836238),
        ('c4', -3552.866100005229)
    ])

def test_predict_single_no_prior_no_return_scores():
    """Predict a single observation sequence with no prior and no returned scores"""
    prediction = hmm_clf.predict(x, prior=False, return_scores=False)
    assert prediction == 'c0'

def test_predict_multiple_with_prior_with_return_scores():
    """Predict multiple observation sequences with a prior and returned scores"""
    predictions = hmm_clf.predict(X, prior=True, return_scores=True)
    assert predictions == [
        ('c0', [
            ('c0', -3587.3909579390056),
            ('c1', -3532.9727085623044),
            ('c2', -3532.837286683744),
            ('c3', -3550.3438107711895),
            ('c4', -3551.2566620927946)
        ]),
        ('c0', [
            ('c0', -3673.214015309591),
            ('c1', -3618.434054039467),
            ('c2', -3618.260500921377),
            ('c3', -3635.992529296514),
            ('c4', -3636.7415856020866)
        ]),
        ('c0', [
            ('c0', -3758.797608265826),
            ('c1', -3703.662416915794),
            ('c2', -3703.469395737211),
            ('c3', -3721.4453332093794),
            ('c4', -3722.0405954774333)
        ])
    ]

def test_predict_multiple_with_prior_no_return_scores():
    """Predict multiple observation sequences with a prior and no returned scores"""
    predictions = hmm_clf.predict(X, prior=True, return_scores=False)
    assert predictions == ['c0', 'c0', 'c0']

def test_predict_multiple_no_prior_with_return_scores():
    """Predict multiple observation sequences with no prior and returned scores"""
    predictions = hmm_clf.predict(X, prior=False, return_scores=True)
    assert predictions == [
        ('c0', [
            ('c0', -3589.00039585144),
            ('c1', -3534.5821464747387),
            ('c2', -3534.4467245961782),
            ('c3', -3551.9532486836238),
            ('c4', -3552.866100005229)
        ]),
        ('c0', [
            ('c0', -3674.8234532220254),
            ('c1', -3620.043491951901),
            ('c2', -3619.8699388338114),
            ('c3', -3637.6019672089483),
            ('c4', -3638.351023514521)
        ]),
        ('c0', [
            ('c0', -3760.4070461782603),
            ('c1', -3705.2718548282282),
            ('c2', -3705.0788336496453),
            ('c3', -3723.0547711218137),
            ('c4', -3723.6500333898675)
        ])
    ]

def test_predict_single_no_prior_no_return_scores():
    """Predict multiple observation sequences with no prior and no returned scores"""
    predictions = hmm_clf.predict(X, prior=False, return_scores=False)
    assert predictions == ['c0', 'c0', 'c0']

# ======================== #
# HMMClassifier.evaluate() #
# ======================== #

def test_evaluate_with_prior_with_labels():
    """Evaluate with a prior and confusion matrix labels"""
    acc, cm = hmm_clf.evaluate(X, Y, prior=True, labels=['c0', 'c1', 'c2', 'c3', 'c4'])
    assert_equal(acc, 0.3333333333333333)
    assert_equal(cm, np.array([
        [1, 0, 0, 0, 0],
        [2, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ]))

def test_evaluate_with_prior_no_labels():
    """Evaluate with a prior and no confusion matrix labels"""
    acc, cm = hmm_clf.evaluate(X, Y, prior=True, labels=None)
    assert_equal(acc, 0.3333333333333333)
    assert_equal(cm, np.array([
        [1, 0],
        [2, 0]
    ]))

def test_evaluate_no_prior_with_labels():
    """Evaluate with no prior and confusion matrix labels"""
    acc, cm = hmm_clf.evaluate(X, Y, prior=False, labels=['c0', 'c1', 'c2', 'c3', 'c4'])
    assert_equal(acc, 0.3333333333333333)
    assert_equal(cm, np.array([
        [1, 0, 0, 0, 0],
        [2, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ]))

def test_evaluate_no_prior_no_labels():
    """Evaluate with no prior and no confusion matrix labels"""
    acc, cm = hmm_clf.evaluate(X, Y, prior=False, labels=None)
    assert_equal(acc, 0.3333333333333333)
    assert_equal(cm, np.array([
        [1, 0],
        [2, 0]
    ]))