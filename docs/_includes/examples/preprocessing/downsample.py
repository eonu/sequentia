import numpy as np
from sequentia.preprocessing import downsample

# Create some sample data
X = [np.random.random((10 * i, 3)) for i in range(1, 4)]

# Downsample the data with downsample factor 5 and decimation
X = downsample(X, n=5, method='decimate')