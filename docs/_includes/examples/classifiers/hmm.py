import numpy as np
from sequentia.classifiers import HMM

# Create some sample data
X = [np.random.random((10 * i, 3)) for i in range(1, 4)]

# Create and fit a left-right HMM with random transitions and initial state distribution
hmm = HMM(label='class1', n_states=5, topology='left-right')
hmm.set_random_initial()
hmm.set_random_transitions()
hmm.fit(X)