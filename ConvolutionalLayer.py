import numpy as np
from scipy import signal


class ConvolutionalLayer:
    # initialize input tensor and kernel shape
    def __init__(self, shape, size_k, depth_k):
        # store input parameters
        # height, width, and depth of input
        self.shape = shape
        # number of kernels
        self.depth_k = depth_k
        # the size of each matrix within each kernel
        self.size_k = size_k

        # format output
        self.shape_out = (depth_k,
                          shape[0] - size_k + 1,
                          shape[1] - size_k + 1)

        # format kernels
        self.shape_k = (depth_k, self.shape[2], size_k, size_k)
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
            for j in range(self.shape[2]):
                # cross-correlate the inputs and the kernels
                self.output[i] += signal.correlate2d(inputs[j],
                                                   self.kernels[i, j],
                                                   "valid")

    def backward(self, dvalues, lr):
        # initialize gradient structure
        dkernels = np.zeros(self.shape_k)
        dinputs = np.zeros(self.shape)

        # compute gradients
        for i in range(self.depth_k):
            for j in range(self.shape[2]):
                # cross correlate the input tensor with the output gradient
                dkernels[i, j] = signal.correlate2d(self.inputs[j], dvalues[i], "valid")
                # fully convolve the output gradient with the kernel
                dinputs[j] = signal.convolve2d(dvalues[i], self.kernels[i, j], "full")

        # update the kernels and biases
        self.kernels -= lr * dkernels
        self.biases -= lr * dvalues
        