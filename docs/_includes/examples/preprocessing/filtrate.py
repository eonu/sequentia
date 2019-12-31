import numpy as np
from sequentia.preprocessing import filtrate

# Create some sample data
X = [np.random.random((10 * i, 3)) for i in range(1, 4)]

# Filter the data with window size 5 and median filtering
X = filtrate(X, n=5, method='median')