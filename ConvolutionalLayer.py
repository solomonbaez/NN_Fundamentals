import numpy as np
from scipy import signal


# convert input data to 3D tensor
def reshape_features(X):
    return X.reshape(len(X), 1, 28, 28)


class ConvolutionalLayer:
    # initialize input tensor and kernel shape
    def __init__(self, shape, size_k, depth_k):
        # store input parameters
        # depth, height, width of the input
        self.shape = shape
        # number of kernels
        self.depth_k = depth_k
        # the size of each matrix within each kernel
        self.size_k = size_k

        # format output
        self.shape_out = (depth_k,
                          shape[1] - size_k + 1,
                          shape[2] - size_k + 1)

        # format kernels
        self.shape_k = (depth_k, self.shape[0], size_k, size_k)
        self.kernels = np.random.randn(*self.shape_k)

        # format biases
        self.biases = np.random.randn(*self.shape_out)

    def forward(self, inputs):
        # store inputs
        self.inputs = inputs

        # initialize output array
        self.output = np.copy(self.biases)

        # calculate output
        # loop through layer depth
        for i in range(self.depth_k):
            for j in range(self.shape[0]):
                # cross-correlate the inputs and the kernels
                self.output[i] += \
                    signal.correlate2d(inputs[j], self.kernels[i, j], "valid")

    def backward(self, dvalues):
        # initialize gradient structure
        dkernels = np.zeros(self.shape_k)
        dinputs = np.zeros(self.shape)

        # compute gradients
        for i in range(self.depth_k):
            for j in range(self.shape[0]):
                # cross correlate the input tensor with the output gradient
                dkernels[i, j] = signal.correlate2d(self.inputs[j], dvalues[i], "valid")
                # fully convolve the output gradient with the kernel
                dinputs[j] = signal.convolve2d(dvalues[i], self.kernels[i, j], "full")


# reshape convoluted outputs
class ReshapeLayer:
    def __init__(self, shape_in, shape_out):
        self.shape_in = shape_in
        self.shape_out = shape_out

    # reshape the input to the output shape
    def forward(self, inputs):
        return np.reshape(inputs, self.shape_out)

    # reshape the output to the input shape
    def backward(self, dvalues):
        return np.reshape(dvalues, self.shape_in)


# basic convolutional neural network optimizer
class OptimizerCNN:

    # initialize optimizer: genearlly LR @ 1e-3, decay @ 1e-4
    def __init__(self, learning_rate=1e-3, decay=1e-4):
        self.lr = learning_rate
        self.current_lr = learning_rate
        self.decay = decay
        self.iterations = 0

    # learning rate decay
    def pre_update(self):
        if self.decay:
            self.current_lr = self.lr / (1 + self.decay * self.iterations)

    # update layer parameters
    def update(self, layer):
        # update kernels and biases
        layer.kernels -= self.current_lr * layer.dkernels
        layer.biases -= self.current_lr * layer.dvalues

    # update internal loop
    def post_update(self):
        self.iterations += 1
