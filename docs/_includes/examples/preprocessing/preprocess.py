import numpy as np
from sequentia.preprocessing import Preprocess

# Create some sample data
X = [np.random.random((10 * i, 3)) for i in range(1, 4)]

# Create the Preprocess object
pre = Preprocess()
pre.normalize()
pre.filtrate(n=10, method='median')
pre.downsample(n=5, method='decimate')
pre.fft()

# Transform the data applying transformations in order
X = pre.transform(X)