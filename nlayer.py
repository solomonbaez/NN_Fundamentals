import numpy as np

# import training data from nnfs
import nnfs
from nnfs.datasets import spiral_data

# maintain repeatability
np.random.seed(0)

nnfs.init()

# best practice in ML denotes input features by an X
# in this case, the data is not normalized or scaled
X, y = spiral_data(100, 3)

# generally, you will either construct or load a model
# initial values should be generally be nonzero between (-1, 1)
class Layer_D:
    def __init__(self, n_in, n_neurons):
        # init input size by matrix size, no transposition
        self.weights = 0.1*np.random.randn(n_in, n_neurons)
        # requires a tuple of the shape
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        # perform the output calculation and store
        self.output = np.dot(inputs, self.weights) + self.biases

# rectified linear unit activation function
class ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

# construct the primary layer to process input data
layer_1 = Layer_D(2, 5)
layer_1.forward(X)

# layer activation using the ReLU model
activation1 = ReLU()
activation1.forward(layer_1.output)
