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
        self.weights = 0.01 * np.random.randn(n_in, n_neurons)
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
        self.inputs = inputs

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

# Categorical Cross Entropy loss function
class LossCCE:
    # SoftMax processed data is passed as the input, y as the target
    def forward(self, inputs, targets):
        samples = len(inputs)

        # prevent /0 process death
        clipped_inputs = np.clip(inputs, 1e-7, 1 - 1e-7)

        # ensure processing of both scalar and one-hot encoded inputs
        if len(targets.shape) == 1:
            confidences = clipped_inputs[range(samples), targets]
        elif len(targets.shape) == 2:
            confidences = np.sum(clipped_inputs * targets, axis=1)

        # calculate and return CCE data loss
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
        self.activation = SoftMax()
        self.loss = LossCCE()

    # the forward method will calculate the SoftMax and CCE
    def forward(self, inputs, targets):
        self.activation.forward(inputs)
        self.output = self.activation.output
        # return the loss for variable storage
        return self.loss.forward(self.output, targets)

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

# SGD optimizer
class OptimizerSGD:

    # initialize optimizer
    # learning rate set to 1.0 as default
    def __init__(self, learning_rate=1.0):
        self.lr = learning_rate

    # update layer parameters using optimizer settings
    def update(self, layer):
        layer.weights += -self.lr * layer.dweights
        layer.biases += -self.lr * layer.dbiases

# primary layer, 2 input features, 64 outputs
layer1 = DeepLayer(2, 64)

# primary layer activation using the ReLU method
activation1 = ReLU()

# secondary layer, 64 input features, 3 classes
layer2 = DeepLayer(64, 3)

# SoftMax classifier's combined activation and CCE loss
cbp = CombinedBP()

# optimizer initialization
optimizer = OptimizerSGD(learning_rate=1)

# train the model
for epoch in range(10001):
    # forward passes on the first layer/activation pair
    layer1.forward(X)
    activation1.forward(layer1.output)

    # forward pass on layer two
    layer2.forward(activation1.output)

    # activation/loss on layer two
    activation2 = cbp.forward(layer2.output, y)

    # store activation2 outputs along axis 1
    predictions = np.argmax(cbp.output, axis=1)
    # discretize one-hot encoded targets
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)

    # calculate accuracy from activation2 and targets
    accuracy = np.mean(predictions==y)
    # report epoch performance
    if not epoch % 100:
        print(f"epoch: {epoch}, " +
              f"accuracy: {accuracy:.4f}, " +
              f"loss: {activation2:.4f}")

    # backpropogation, starting at the most recent forward pass
    # combined SoftMax and CCE backwards step implicit to the cbp object
    cbp.backward(cbp.output, y)
    # reverse sequential utilization of layer/activation gradients
    layer2.backward(cbp.dinputs)
    activation1.backward(layer2.dinputs)
    layer1.backward(activation1.dinputs)

    # update weights and biases in the model layers
    optimizer.update(layer1)
    optimizer.update(layer2)