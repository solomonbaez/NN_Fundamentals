import numpy as np

# inputs fed into the neuron may be neuronal outputs or external data
# Type: 1D array, Vector
inputs = [1, 2, 3, 2.5]

# weight is unique to each neuron
# linearly mutates the input values using a constant
# Type: 2D array, Matrix
weights = [[0.4, 0.2, -0.6, 1.0],
           [0.2, 0.10, 0.6, -1.0],
           [-0.48, 2, 0.53, 0.67]]

# bias is distinct and unique to each neuron
# offsets the input values using a constant
# Type: 1D array, Vector
biases = [2, 3, 0.5]

# weights area called before inputs in the dot product because we want
# the index to be set to the 2D array of neurons, not the inputs
output = np.dot(weights, inputs) + biases

print(output)