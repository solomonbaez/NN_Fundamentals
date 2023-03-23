import numpy as np
import matplotlib.pyplot as plt

# import and initialize training data from nnfs
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()


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

# common loss class
class Loss:
    # calculate data and regulazation loss
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)

        # return data loss
        return np.mean(sample_losses)


# Categorical Cross Entropy loss function
class LossCCE(Loss):
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
        losses = -np.log(confidences)
        return losses


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
        return self.loss.calculate(self.output, targets)

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

    # initialize optimizer: LR @ 1.0, decay @ 1e-1
    def __init__(self, learning_rate=1.0, decay=0.0, momentum=0.0):
        self.lr = learning_rate
        self.current_lr = learning_rate
        self.decay = decay
        self.momentum = momentum
        self.iterations = 0

    # learning rate decay
    def lr_decay(self, epoch):
        if self.decay:
            self.current_lr = self.lr / (1 + self.decay * epoch)

    # update layer parameters using optimizer settings
    def update(self, layer):
        # determine if momentum is utilized
        if self.momentum:

            # if absent, create momentum arrays within the layer
            if not hasattr(layer, 'momentum_w'):
                layer.momentum_w = np.zeros_like(layer.weights)
                layer.momentum_b = np.zeros_like(layer.biases)

            # calculate weight updates with momentum
            update_w = self.momentum * layer.momentum_w \
                            - self.current_lr * layer.dweights
            layer.momentum_w = update_w

            # calculate bias updates
            update_b = self.momentum * layer.momentum_b \
                            - self.current_lr * layer.dbiases
            layer.momentum_b = update_b

        # standard SGD, no momentum
        else:
            update_w = -self.current_lr * layer.dweights
            update_b = -self.current_lr * layer.dbiases

        # update weights and biases
        layer.weights += update_w
        layer.biases += update_b

# Adaptive Gradient optimizer
class OptimizerADA:

    # initialize optimizer
    def __init__(self, learning_rate=1.0, decay=0.0, epsilon = 1e-7):
        self.lr = learning_rate
        self.current_lr = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.iterations = 0

    # learning rate decay
    def lr_decay(self, epoch):
        if self.decay:
            self.current_lr = self.lr / (1 + self.decay * epoch)

    # update layer parameters using optimizer settings
    def update(self, layer):

        # if absent, create cache arrays within the layer
        if not hasattr(layer, 'cache_w'):
            layer.cache_w = np.zeros_like(layer.weights)
            layer.cache_b = np.zeros_like(layer.biases)

        # update chache arrays with current gradients
        layer.cache_w += layer.dweights**2
        layer.cache_b += layer.dbiases**2

        # update weights and biases
        layer.weights += -self.current_lr * layer.dweights / \
                         (np.sqrt(layer.cache_w) + self.epsilon)
        layer.biases += -self.current_lr * layer.dbiases / \
                         (np.sqrt(layer.cache_b) + self.epsilon)

# Root Mean Square Propogation optimizer
class OptimizerRMS:

    # initialize optimizer
    # gradient momentum is maintained so learning rate must be truncated
    def __init__(self, learning_rate=0.001, decay=0.0, epsilon = 1e-7, rho=0.9):
        self.lr = learning_rate
        self.current_lr = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.rho = rho
        self.iterations = 0

    # learning rate decay
    def lr_decay(self, epoch):
        if self.decay:
            self.current_lr = self.lr / (1 + self.decay * epoch)

    # update layer parameters using optimizer settings
    def update(self, layer):

        # if absent, create cache arrays within the layer
        if not hasattr(layer, 'cache_w'):
            layer.cache_w = np.zeros_like(layer.weights)
            layer.cache_b = np.zeros_like(layer.biases)

        # update chache arrays with squared current gradients
        layer.cache_w = self.rho * layer.cache_w + (1 - self.rho) \
                         * layer.dweights**2
        layer.cache_b = self.rho * layer.cache_b + (1 - self.rho) \
                         * layer.dbiases**2

        # update weights and biases
        layer.weights += -self.current_lr * layer.dweights / \
                         (np.sqrt(layer.cache_w) + self.epsilon)
        layer.biases += -self.current_lr * layer.dbiases / \
                         (np.sqrt(layer.cache_b) + self.epsilon)

# Root Mean Square Propogation optimizer
class OptimizerAdaM:

    # initialize optimizer: genearlly LR @ 1e-3, decay @ 1e-4
    # gradient momentum is maintained so learning rate must be truncated
    def __init__(self, learning_rate=0.001, decay=0.0, epsilon = 1e-7, beta1=0.9, beta2=0.999):
        self.lr = learning_rate
        self.current_lr = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.iterations = 0

    # learning rate decay
    def lr_decay(self, epoch):
        if self.decay:
            self.current_lr = self.lr / (1 + self.decay * epoch)

    # update layer parameters using optimizer settings
    def update(self, layer):

        # if absent, create cache arrays within the layer
        if not hasattr(layer, 'cache_w'):
            layer.momentum_w = np.zeros_like(layer.weights)
            layer.cache_w = np.zeros_like(layer.weights)
            layer.momentum_b = np.zeros_like(layer.biases)
            layer.cache_b = np.zeros_like(layer.biases)

        # update momentums with current gradients
        layer.momentum_w = self.beta1 * layer.momentum_w + (1 - self.beta1) \
                         * layer.dweights
        layer.momentum_b = self.beta1 * layer.momentum_b + (1 - self.beta1) \
                         * layer.dbiases

        # correct momentum, starting at 1
        momentum_w_corrected = layer.momentum_w / \
                               (1 - self.beta1 ** (self.iterations + 1))
        momentum_b_corrected = layer.momentum_b / \
                               (1 - self.beta1 ** (self.iterations + 1))

        # update caches with squared current gradients
        layer.cache_w = self.beta2 * layer.cache_w + (1 - self.beta2) \
                        * layer.dweights ** 2
        layer.cache_b = self.beta2 * layer.cache_b + (1 - self.beta2) \
                        * layer.dbiases ** 2

        # correct caches
        cache_w_corrected = layer.cache_w / \
                            (1 - self.beta2 ** (self.iterations + 1))
        cache_b_corrected = layer.cache_b / \
                            (1 - self.beta2 ** (self.iterations + 1))

        # update weights and biases
        layer.weights += -self.current_lr * momentum_w_corrected / \
                         (np.sqrt(cache_w_corrected) + self.epsilon)
        layer.biases += -self.current_lr * momentum_b_corrected / \
                        (np.sqrt(cache_b_corrected) + self.epsilon)

# create dataset
X, y = spiral_data(samples=100, classes=3)

# primary layer, 2 input features, 64 outputs
layer1 = DeepLayer(2, 64)

# primary layer activation using the ReLU method
activation1 = ReLU()

# secondary layer, 64 input features, 3 classes
layer2 = DeepLayer(64, 3)

# SoftMax classifier's combined activation and CCE loss
cbp = CombinedBP()

# optimizer initialization
optimizer = OptimizerAdaM(learning_rate=0.05, decay=5e-7)

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
              f"accuracy: {accuracy:.3f}, " +
              f"loss: {activation2:.3f}, " +
              f"lr: {optimizer.current_lr}")

    # backpropogation, starting at the most recent forward pass
    # combined SoftMax and CCE backwards step implicit to the cbp object
    cbp.backward(cbp.output, y)
    # reverse sequential utilization of layer/activation gradients
    layer2.backward(cbp.dinputs)
    activation1.backward(layer2.dinputs)
    layer1.backward(activation1.dinputs)

    # decay the learning rate
    optimizer.lr_decay(epoch)
    # update weights and biases in the model layers
    optimizer.update(layer1)
    optimizer.update(layer2)