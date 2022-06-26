import os, pickle, pytest, warnings, os, numpy as np, hmmlearn.hmm
from copy import deepcopy
from sequentia.classifiers import GMMHMM, HMMClassifier, _ErgodicTopology
from sequentia.datasets import load_random_sequences
from ....support import assert_equal, assert_not_equal

# pytest.skip('Skip until datasets module is added and positive definite issues are fixed', allow_module_level=True)

# Set seed for reproducible randomness
random_state = np.random.RandomState(0)

# Create some sample data
dataset = load_random_sequences(50, n_features=2, n_classes=5, length_range=(10, 30), random_state=random_state)
dataset.classes = [f'c{i}' for i in range(5)]
dataset.y = np.array([f'c{i}' for i in dataset.y])
x, y = dataset[0]

# Create and fit some HMMs
hmms = []
for sequences, label in dataset.iter_by_class():
    hmm = GMMHMM(label=label, n_states=5, random_state=random_state)
    hmm.set_random_initial()
    hmm.set_random_transitions()
    hmm.fit(sequences)
    hmms.append(hmm)

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
    assert prediction == 1

def test_predict_single_uniform_prior():
    """Predict a single observation sequence with a uniform prior"""
    prediction = hmm_clf.predict(x, prior='uniform', return_scores=False, original_labels=False)
    assert prediction == 1

def test_predict_single_custom_prior():
    """Predict a single observation sequence with a custom prior"""
    prediction = hmm_clf.predict(x, prior=([1e-50]*4+[1-4e-50]), return_scores=False, original_labels=False)
    assert prediction == 1

def test_predict_single_return_scores():
    """Predict a single observation sequence and return the transformed label, with the un-normalized posterior scores"""
    prediction = hmm_clf.predict(x, prior='frequency', return_scores=True, original_labels=False)
    assert isinstance(prediction, tuple)
    assert prediction[0] == 1
    assert_equal(prediction[1], np.array([
        -131.46105165, -78.80931343, -99.35179093, -90.89464994, -483.92229446
    ]))

def test_predict_single_original_labels():
    """Predict a single observation sequence and return the original label, without the un-normalized posterior scores"""
    prediction = hmm_clf.predict(x, prior='uniform', return_scores=False, original_labels=True)
    assert prediction == 'c1'

def test_predict_single_return_scores_original_labels():
    """Predict a single observation sequence and return the original label, with the un-normalized posterior scores"""
    prediction = hmm_clf.predict(x, prior='frequency', return_scores=True, original_labels=True)
    assert isinstance(prediction, tuple)
    assert prediction[0] == 'c1'
    assert_equal(prediction[1], np.array([
        -131.46105165, -78.80931343, -99.35179093, -90.89464994, -483.92229446
    ]))

def test_predict_multiple_frequency_prior():
    """Predict multiple observation sequences with a frequency prior"""
    predictions = hmm_clf.predict(dataset.X, prior='frequency', return_scores=False, original_labels=False)
    assert_equal(predictions, np.array([
        1, 3, 1, 0, 1, 3, 1, 1, 3, 2, 1, 2, 3, 2, 4, 1, 3, 2, 0, 0, 1, 1,
        3, 1, 1, 3, 1, 1, 1, 1, 4, 3, 0, 3, 1, 2, 3, 1, 2, 1, 1, 1, 3, 3,
        2, 3, 1, 1, 4, 1
    ]))

def test_predict_multiple_uniform_prior():
    """Predict multiple observation sequences with a uniform prior"""
    predictions = hmm_clf.predict(dataset.X, prior='uniform', return_scores=False, original_labels=False)
    assert_equal(predictions, np.array([
        1, 3, 1, 0, 1, 3, 1, 1, 3, 2, 1, 2, 3, 2, 4, 1, 3, 2, 0, 0, 1, 1,
        3, 1, 1, 3, 1, 1, 1, 1, 4, 3, 0, 3, 1, 2, 3, 1, 2, 1, 1, 1, 3, 3,
        2, 3, 1, 1, 4, 1
    ]))

def test_predict_multiple_custom_prior():
    """Predict multiple observation sequences with a custom prior"""
    predictions = hmm_clf.predict(dataset.X, prior=([1-4e-50]+[1e-50]*4), return_scores=False, original_labels=False)
    assert_equal(predictions, np.array([
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0
    ]))

def test_predict_multiple_return_scores():
    """Predict multiple observation sequences and return the transformed labels, with the un-normalized posterior scores"""
    predictions = hmm_clf.predict(dataset.X, prior='frequency', return_scores=True, original_labels=False)
    assert isinstance(predictions, tuple)
    assert_equal(predictions[0], np.array([
        1, 3, 1, 0, 1, 3, 1, 1, 3, 2, 1, 2, 3, 2, 4, 1, 3, 2, 0, 0, 1, 1,
        3, 1, 1, 3, 1, 1, 1, 1, 4, 3, 0, 3, 1, 2, 3, 1, 2, 1, 1, 1, 3, 3,
        2, 3, 1, 1, 4, 1
    ]))
    assert_equal(predictions[1][:5], np.array([
        [-131.46105165,  -78.80931343,  -99.35179093,  -90.89464994, -483.92229446],
        [ -91.58935678,  -66.6556658 ,  -91.46883547,  -65.69934269, -716.797869  ],
        [ -97.5230626 ,  -74.50878143,  -99.1544397 ,  -76.48361176, -690.2988915 ],
        [  14.24986519,  -44.85298283,  -41.50143234,  -40.50844881, -148.67734234],
        [ -95.11368472,  -40.81069058,  -59.46841129,  -52.60034218, -430.36823963]
    ]))

def test_predict_multiple_original_labels():
    """Predict multiple observation sequences and return the original labels, without the un-normalized posterior scores"""
    predictions = hmm_clf.predict(dataset.X, prior='frequency', return_scores=False, original_labels=True)
    assert all(np.equal(
        predictions.astype(object),
        np.array([
            'c1', 'c3', 'c1', 'c0', 'c1', 'c3', 'c1', 'c1', 'c3', 'c2', 'c1',
            'c2', 'c3', 'c2', 'c4', 'c1', 'c3', 'c2', 'c0', 'c0', 'c1', 'c1',
            'c3', 'c1', 'c1', 'c3', 'c1', 'c1', 'c1', 'c1', 'c4', 'c3', 'c0',
            'c3', 'c1', 'c2', 'c3', 'c1', 'c2', 'c1', 'c1', 'c1', 'c3', 'c3',
            'c2', 'c3', 'c1', 'c1', 'c4', 'c1'
        ], dtype=object)
    ))

def test_predict_multiple_return_scores_original_labels():
    """Predict multiple observation sequences and return the original labels, with the un-normalized posterior scores"""
    predictions = hmm_clf.predict(dataset.X, prior='frequency', return_scores=True, original_labels=True)
    assert isinstance(predictions, tuple)
    assert all(np.equal(
        predictions[0].astype(object),
        np.array([
            'c1', 'c3', 'c1', 'c0', 'c1', 'c3', 'c1', 'c1', 'c3', 'c2', 'c1',
            'c2', 'c3', 'c2', 'c4', 'c1', 'c3', 'c2', 'c0', 'c0', 'c1', 'c1',
            'c3', 'c1', 'c1', 'c3', 'c1', 'c1', 'c1', 'c1', 'c4', 'c3', 'c0',
            'c3', 'c1', 'c2', 'c3', 'c1', 'c2', 'c1', 'c1', 'c1', 'c3', 'c3',
            'c2', 'c3', 'c1', 'c1', 'c4', 'c1'
        ], dtype=object)
    ))
    assert_equal(predictions[1][:5], np.array([
        [-131.46105165,  -78.80931343,  -99.35179093,  -90.89464994, -483.92229446],
        [ -91.58935678,  -66.6556658 ,  -91.46883547,  -65.69934269, -716.797869  ],
        [ -97.5230626 ,  -74.50878143,  -99.1544397 ,  -76.48361176, -690.2988915 ],
        [  14.24986519,  -44.85298283,  -41.50143234,  -40.50844881, -148.67734234],
        [ -95.11368472,  -40.81069058,  -59.46841129,  -52.60034218, -430.36823963]
    ]))

# ======================== #
# HMMClassifier.evaluate() #
# ======================== #

def test_evaluate():
    """Evaluate performance on some observation sequences and labels"""
    acc, cm = hmm_clf.evaluate(dataset.X, dataset.y, prior='frequency')
    assert acc == 0.92
    print(repr(cm))
    assert_equal(cm, np.array([
        [ 4,  0,  0,  0,  0],
        [ 0, 20,  0,  1,  0],
        [ 0,  1,  7,  0,  0],
        [ 0,  2,  0, 12,  0],
        [ 0,  0,  0,  0,  3]
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
        assert [model.label for model in clf.models_] == dataset.classes
        assert list(clf.encoder_.classes_) == dataset.classes
        predictions = clf.predict(dataset.X, prior='frequency', return_scores=True, original_labels=True)
        assert isinstance(predictions, tuple)
        assert all(np.equal(
            predictions[0].astype(object),
            np.array([
                'c1', 'c3', 'c1', 'c0', 'c1', 'c3', 'c1', 'c1', 'c3', 'c2', 'c1',
                'c2', 'c3', 'c2', 'c4', 'c1', 'c3', 'c2', 'c0', 'c0', 'c1', 'c1',
                'c3', 'c1', 'c1', 'c3', 'c1', 'c1', 'c1', 'c1', 'c4', 'c3', 'c0',
                'c3', 'c1', 'c2', 'c3', 'c1', 'c2', 'c1', 'c1', 'c1', 'c3', 'c3',
                'c2', 'c3', 'c1', 'c1', 'c4', 'c1'
            ], dtype=object)
        ))
        assert_equal(predictions[1][:5], np.array([
            [-131.46105165,  -78.80931343,  -99.35179093,  -90.89464994, -483.92229446],
            [ -91.58935678,  -66.6556658 ,  -91.46883547,  -65.69934269, -716.797869  ],
            [ -97.5230626 ,  -74.50878143,  -99.1544397 ,  -76.48361176, -690.2988915 ],
            [  14.24986519,  -44.85298283,  -41.50143234,  -40.50844881, -148.67734234],
            [ -95.11368472,  -40.81069058,  -59.46841129,  -52.60034218, -430.36823963]
        ]))
    finally:
        os.remove('test.pkl')