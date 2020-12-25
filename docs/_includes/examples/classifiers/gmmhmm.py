import numpy as np
from sequentia.classifiers import GMMHMM

# Create some sample data
X = [np.random.random((10 * i, 3)) for i in range(1, 4)]

# Create and fit a left-right HMM with random transitions and initial state distribution
hmm = GMMHMM(label='class1', n_states=10, n_components=3, topology='left-right', covariance_type='diag')
hmm.set_random_initial()
hmm.set_random_transitions()
hmm.fit(X)