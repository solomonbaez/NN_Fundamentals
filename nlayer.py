import numpy as np
import matplotlib.pyplot as plt

# import training data from nnfs
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# initialize training dataset
X, y = spiral_data(samples=100, classes=3)

# generally, you will either construct or load a model
# initial values should be generally be nonzero between (-1, 1)
class DeepLayer:
    def __init__(self, n_in, n_neurons):
        # init input size by matrix size, no transposition
        self.weights = 0.1*np.random.randn(n_in, n_neurons)
        # requires a tuple of the shape
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        # perform the output calculation and store
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        # Backpropogation gradient production
        # Weight gradient component
        self.dweights = np.dot(self.inputs.T, dvalues)
        # Bias gradient component
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Input gradient component
        self.dinputs = np.dot(dvalues, self.weights.T)

# Rectified Linear Unit activation function
class ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        # Copy the input gradient values and structure
        self.dinputs = dvalues.copy()
        # Produce a zero gradient where input values were invalid
        self.dinputs[self.inputs <= 0] = 0

# SoftMax activation function
class SoftMax:
    def forward(self, inputs):
        # exponential function application and normalization by the maximum input
        exponentials = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # normalization per softmax methodology
        self.output = exponentials/np.sum(exponentials, axis=1, keepdims=True)

# Categorical Cross Entropy function, conceptual implementation
class CCE:
    def calculate(self, inputs, targets):
        samples = len(inputs)

        # prevent zero induced process death
        clipped_targets = np.clip(inputs, 1e-7, 1-1e-7)

        # ensure processing of both scalar and one-hot encoded inputs
        if len(targets.shape) == 1:
            confidences = clipped_targets[range(samples), targets]
        elif len(targets.shape) == 2:
            confidences = np.sum(clipped_targets*targets, axis=1)

        # calculate and return CCE
        loss = -np.log(confidences)
        return np.mean(loss)

# construct the primary layer to process input data
layer_1 = DeepLayer(2, 3)
layer_1.forward(X)

# layer activation using the ReLU method
activation1 = ReLU()
activation1.forward(layer_1.output)

# second layer, three classes
layer_2 = DeepLayer(3, 3)
layer_2.forward(activation1.output)

# second layer activation using the SoftMax method
# probability distribution values are consistent at approximately 0.33
# expected due to the random initialization of the model
activation2 = SoftMax()
activation2.forward(layer_2.output)

# Categorical Cross Entropy loss calculation
# SoftMax processed data is passed as the input, y as the target
cce = CCE()
loss = cce.calculate(activation2.output, y)

print(loss)


