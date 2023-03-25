import numpy as np

# generally, you will either construct or load a model
# initial values should be generally be nonzero between (-1, 1)
class DeepLayer:
    def __init__(self, n_in, n_neurons, l1_w=0.0, l2_w=0.0, l1_b=0.0, l2_b=0.0):
        # initialize input size
        self.weights = 0.01 * np.random.randn(n_in, n_neurons)
        self.biases = np.zeros((1, n_neurons))

        # regularization strength
        self.l1_w = l1_w
        self.l2_w = l2_w
        self.l1_b = l1_b
        self.l2_b = l2_b

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

        # L1 and L2 backpropogation on weights
        if self.l1_w > 0:
            dl1 = np.ones_like(self.weights)
            dl1[self.weights < 0] = -1
            self.dweights += self.l1_w * dl1

        if self.l2_w > 0:
            self.dweights += 2 * self.l2_w * self.weights

        # L1 and L2 backpropogation on biases
        if self.l1_b > 0:
            dl1 = np.ones_like(self.biases)
            dl1[self.biases < 0] = -1
            self.dbiases += self.l1_b * dl1

        if self.l2_b > 0:
            self.dbiases += 2 * self.l2_b * self.biases

        # input gradient component
        self.dinputs = np.dot(dvalues, self.weights.T)

# Dropout Layer class used exclusively in training
# randomly disable neurons (sets outputs to zero) at a given rate per forward pass
# forces more neurons to learn the data
# increases the likelyhood of understanding the underlying function in a dataset
class DropoutLayer:
    def __init__(self, rate):
        # rate is inverted and stored
        self.rate = 1 - rate

    # Bernoulli disribution filter
    # P (r = 1) = p, P (r = 0) = q
    # where q = 1 - p and q == ratio of neurons to disable
    # importantly, output values must be scaled to match training/prediction states
    def forward(self, inputs):
        self.inputs = inputs

        # generate a scaled mask
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / \
                           self.rate
        # apply mask to inputs
        self.output = inputs * self.binary_mask

    # partial derivative of Bernoulli distribution
    # f'(r = 0) = 0, f'(r > 0) = (r)(1-q)**(-1)
    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask

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


# Sigmoid activation function
# first component for binary logistic regression
# single neurons can represent two classes
class Sigmoid:
    def forward(self, inputs):
        self.inputs = inputs

        # Sigmoid activation
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues):
        # Partial derivative of the sigmoid activation function
        # s' = s(1 - s)
        self.dinputs = dvalues * (1 - self.output) * self.output


# common loss class
class Loss:
    # calculate data and regulazation loss
    def regularization(self, layer):
        # initialize regularization loss
        loss_r = 0

        # add l1 and l2 regularization to the temporary loss
        if layer.l1_w > 0:
            loss_r += layer.l1_w * np.sum(np.abs(layer.weights))

        if layer.l2_w > 0:
            loss_r += layer.l2_w * np.sum(layer.weights * layer.weights)

        if layer.l1_b > 0:
            loss_r += layer.l1_b * np.sum(np.abs(layer.biases))

        if layer.l2_b > 0:
            loss_r += layer.l2_b * np.sum(layer.biases * layer.biases)

        return loss_r

    def calculate(self, output, y):
        sample_loss = self.forward(output, y)

        return np.mean(sample_loss)


# Categorical Cross Entropy loss function
class LossCCE(Loss):
    # SoftMax processed data is passed as the input, y as the target
    def forward(self, inputs, targets):
        samples = len(inputs)

        # prevent /0 process death without impacting the mean
        clipped_inputs = np.clip(inputs, 1e-7, 1 - 1e-7)

        # ensure processing of both scalar and one-hot encoded inputs
        if len(targets.shape) == 1:
            confidences = clipped_inputs[range(samples), targets]
        elif len(targets.shape) == 2:
            confidences = np.sum(clipped_inputs * targets, axis=1)

        # calculate and return CCE data loss
        losses = -np.log(confidences)
        return losses


# Binary Cross-Entropy loss function
# second component of binary logistic regression
class BinaryCE(Loss):
    # class values are either 0 or 1
    # thus, the incorrect class = 1 - correct class
    def forward(self, inputs, targets):
        clipped_inputs = np.clip(inputs, 1e-7, 1 - 1e-7)

        # loss calculated on a single ouput is a vector of losses
        # sample loss will be the mean of losses from a single sample
        # sample loss = ((current output)**-1) * sum(loss)
        sample_losses = -(targets * np.log(clipped_inputs) + (1 - targets)
                          * np.log(1 - clipped_inputs))

        return np.mean(sample_losses, axis=-1)

    def backward(self, dvalues, targets):
        samples = len(dvalues)
        outputs = len(dvalues[0])

        # prevent /0 process death without impacting the mean
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)

        # calculate and normalize the gradient
        self.dinputs = -(targets / clipped_dvalues -
                         (1 - targets) / (1 - clipped_dvalues)) / outputs
        self.dinputs = self.dinputs / samples


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
