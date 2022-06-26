import librosa
from sequentia.preprocessing import Compose, Custom, Standardize
from sequentia.classifiers import GMMHMM, HMMClassifier
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

# Create and fit a HMM for each class - only training on the sequences that belong to that class
hmms = []
for sequences, label in train_set.iter_by_class():
    # Create a linear HMM with 3 states and 5 components in the GMM emission state distributions
    hmm = GMMHMM(label=label, n_states=3, n_components=5, topology='linear')
    # Set random initial state distributions and transition matrix according to the linear topology
    hmm.set_random_initial()
    hmm.set_random_transitions()
    # Fit each HMM only on the observation sequences which had that label
    hmm.fit(sequences)
    hmms.append(hmm)

# Fit the classifier on the trained HMMs
clf = HMMClassifier().fit(hmms)

# Make a single prediction
y0_pred = clf.predict(test_set.X[0])

# Make multiple predictions
y_pred = clf.predict(test_set.X)

# Make multiple predictions and return class scores (with parallelization)
y_pred, y_pred_scores = clf.predict(test_set.X, return_scores=True, n_jobs=-1)

# Calculate accuracy and generate confusion matrix (with parallelization)
accuracy, confusion = clf.evaluate(test_set.X, test_set.y, n_jobs=-1)