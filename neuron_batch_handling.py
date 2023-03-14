import numpy as np

# inputs fed into the neuron may be neuronal outputs or external data
# Type: 2D array, Matrix
inputs = [[1.0, 2.0, 3.0, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

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

weights2 = [[-0.1, -0.14, 0.5],
            [0.4, 0.12, -0.33],
            [-0.66, 3, 0.01]]

biases2 = [22, 4, 15.5]

# within np.dot, dim 1 must == dim 2
# thus weights must be transposed to conform with the input shape

# multilayer data processing
l1_outputs = np.dot(inputs, np.array(weights).T) + biases
l2_outputs = np.dot(l1_outputs, np.array(weights2).T) + biases2
print(l2_outputs)