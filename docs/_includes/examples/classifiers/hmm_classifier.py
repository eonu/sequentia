import numpy as np
from sequentia.classifiers import HMM, HMMClassifier

# Set of possible labels
labels = [f'class{i}' for i in range(5)]

# Create and fit some sample HMMs
hmms = []
for i, label in enumerate(labels):
    hmm = HMM(label=label, n_states=(i + 3), topology='left-right')
    hmm.set_random_initial()
    hmm.set_random_transitions()
    hmm.fit([np.arange((i + j * 20) * 30).reshape(-1, 3) for j in range(1, 4)])
    hmms.append(hmm)

# Create some sample test data and labels
X = [np.random.random((10 * i, 3)) for i in range(1, 4)]
y = ['class0', 'class1', 'class1']

# Create a classifier and calculate predictions and evaluations
clf = HMMClassifier()
clf.fit(hmms)
predictions = clf.predict(X)
accuracy, confusion = clf.evaluate(X, y, labels=labels)