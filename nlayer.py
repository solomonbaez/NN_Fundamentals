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
        # backpropogation gradient production
        # weight gradient component
        self.dweights = np.dot(self.inputs.T, dvalues)
        # bias gradient component
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # input gradient component
        self.dinputs = np.dot(dvalues, self.weights.T)

# Rectified Linear Unit activation function
class ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        # copy the input gradient values and structure
        self.dinputs = dvalues.copy()
        # produce a zero gradient where input values were invalid
        self.dinputs[self.inputs <= 0] = 0

# SoftMax activation function
class SoftMax:
    def forward(self, inputs):
        # exponential function application and normalization by the maximum input
        exponentials = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # normalization per softmax methodology
        self.output = exponentials/np.sum(exponentials, axis=1, keepdims=True)

    # backpropogation step, obsolete due to CombinedBP class
    def backward(self, dvalues):
        # create an uninitialized array to store sample-wise gradients
        self.dinputs = np.empty_like(dvalues)

        # enumerate outputs and gradients
        for i, (output, dvalue) in enumerate(zip(self.output, dvalues)):
            # flatten the output array
            output = np.array(self.output).reshape(-1, 1)
            # calculate partial derivatives
            # diagflat produces a diagonal array of output values
            jacobian = np.diagflat(output) - np.dot(output, output.T)

            # generate the array of sample-wise gradients
            self.dinputs[i] = np.dot(jacobian, dvalue)

# Categorical Cross Entropy function
class CCE:
    def forward(self, inputs, targets):
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

    # backpropogation step, obsolete due to CombinedBP class
    def backward(self, dvalues, targets):
        samples = len(dvalues)

        # collect labels from the first sample vector
        labels = len(dvalues[0])

        # check if labels are sparse
        if len(targets.shape) == 1:
            # one-hot vectorize the targets
            # (target values on the diagonal, zeroes elsewhere)
            targets = np.eye(labels)[targets]

        # calculate the gradient via the partial derivative of CCE
        # normalize by sample size
        self.dinputs = (-targets/dvalues)/samples

# combined SoftMax and CCE backpropogation class
class CombinedBP:
    def __init__(self):
        self.softmax = SoftMax()
        self.cce = CCE()

    # the forward method will calculate the SoftMax and CCE
    def forward(self, inputs, targets):
        self.softmax.forward(inputs)
        # return the loss for variable storage
        return self.cce.forward(self.softmax.output, targets)

    # the backward method will backpropogate the SoftMax and CCE
    def backward(self, dvalues, targets):
        samples = len(dvalues)

        # discretize one-hot encoded targets
        if len(targets.shape) == 2:
            targets = np.argmax(targets, axis=1)

        self.dinputs = dvalues.copy()
        # calculate backward pass gradient
        self.dinputs[range(samples), targets] -= 1
        # normalize gradient
        self.dinputs /= samples

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
loss = cce.forward(activation2.output, y)

# combined SoftMax and CCE backpropogation
cbp = CombinedBP()
cbp.backward(activation2.output, y)

# display the combined SoftMax and CCE gradients
print(cbp.dinputs)


