import numpy as np
from sequentia.preprocessing import trim_zeros

# Create some sample data
z = np.zeros((4, 3))
x = lambda i: np.vstack((z, np.random.random((10 * i, 3)), z))
X = [x(i) for i in range(1, 4)]

# Zero-trim the data
X = trim_zeros(X)