import numpy as np
import math

# import training data from nnfs
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# initialize training dataset
X, y = spiral_data(samples=100, classes=3)

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


# Rectified Linear Unit activation function
class ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

# SoftMax activation function
class SoftMax:
    def forward(self, inputs):
        # exponential function application and normalization by the maximum input
        exponentials = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # normalization per softmax methodology
        self.output = exponentials/np.sum(exponentials, axis=1, keepdims=True)


# construct the primary layer to process input data
layer_1 = Layer_D(2, 3)
layer_1.forward(X)

# layer activation using the ReLU method
activation1 = ReLU()
activation1.forward(layer_1.output)

# second layer, three classes
layer_2 = Layer_D(3, 3)
layer_2.forward(activation1.output)

# second layer activation using the SoftMax method
activation2 = SoftMax()
activation2.forward(layer_2.output)

# probability distribution values are consistent at approximately 0.33
# expected due to the random initialization of the model
print(activation2.output)

