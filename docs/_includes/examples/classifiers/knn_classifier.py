import numpy as np
from sequentia.classifiers import KNNClassifier

# Create some sample data
X = [np.random.random((10 * i, 3)) for i in range(1, 4)]
y = ['class0', 'class1', 'class1']

# Create and fit the classifier
clf = KNNClassifier(k=1, classes=list(set(y)))
clf.fit(X, y)

# Predict labels for the training data (just as an example)
clf.predict(X)