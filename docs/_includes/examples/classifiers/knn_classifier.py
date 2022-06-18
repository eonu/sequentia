import librosa
from sequentia.preprocessing import Compose, Custom, Standardize
from sequentia.classifiers import KNNClassifier
from sequentia.datasets import load_digits

# Load the FSDD dataset and split into training and testing
dataset = load_digits()
train_set, test_set = dataset.split(split_size=0.2, stratify=True, shuffle=True)

# Set MFCC configuration
spec_kwargs = {'sr': 8000, 'n_mfcc': 5, 'n_fft': 1024, 'hop_length': 256, 'power': 2}

# Create preprocessing pipeline
transforms = Compose([
    Custom(lambda x: librosa.feature.mfcc(x.flatten(), **spec_kwargs).T, name='MFCCs', desc='Generate MFCCs'),
    Standardize()
])

# Apply transformations to the training and test set
train_set.X = transforms(train_set.X)
test_set.X = transforms(test_set.X)

# Create and fit a 1-NN classifier using C-compiled DTW functions
clf = KNNClassifier(k=1, classes=range(10), use_c=True).fit(train_set.X, train_set.y)

# Make a single prediction
y0_pred = clf.predict(test_set.X[0])

# Make multiple predictions
y_pred = clf.predict(test_set.X)

# Make multiple predictions and return class scores (with parallelization)
y_pred, y_pred_scores = clf.predict(test_set.X, return_scores=True, n_jobs=-1)

# Calculate accuracy and generate confusion matrix (with parallelization)
accuracy, confusion = clf.evaluate(test_set.X, test_set.y, n_jobs=-1)