import numpy as np
from sequentia.preprocessing import Preprocess

# Create some sample data
X = [np.random.random((20 * i, 3)) for i in range(1, 4)]

# Create the Preprocess object
pre = Preprocess()
pre.center()
pre.standardize()
pre.filtrate(n=5, method='median')
pre.downsample(n=5, method='decimate')
pre.fft()

# View a summary of the preprocessing steps
pre.summary()

# Transform the data applying transformations in order
X = pre.transform(X)