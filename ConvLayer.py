import numpy as np

# Convolutinal Layer: single kernel functionality
class ConvLayer:
    def __init__(self, shape_in, shape_k):
        # initialize the kernel
        self.kernels = np.random.randn(*shape_k)

        # store batch size and kernel dimensions
        self.size_batch = shape_in[0]
        self.h_k, self.w_k, self.d_k = self.kernels.shape

        # initialize output tensor shape
        self.H = shape_in[1] - self.h_k + 1
        self.W = shape_in[2] - self.w_k + 1
        self.D = self.d_k

        # initialize the bias tensor
        self.biases = np.random.randn(self.H, self.W, self.D)

    def forward(self, inputs):
        # initialize output storage tensor
        self.output = []

        # loop input features through output tensor dimensions
        for idx, x in enumerate(inputs):
            out = np.copy(self.biases)
            for h in range(self.H):
                for w in range(self.W):
                    for d in range(self.D):
                        # extract the current sample
                        sample = x[h:h+self.h_k, w:w+self.w_k, :]

                        # compute output tensor
                        out[h, w, d] = np.sum(sample*self.kernels)

            # store output tensor
            self.output.append(out)

        # restructure the output
        self.output = np.array(self.output)
        self.output = self.output.reshape((self.size_batch, self.H * self.W * self.D))

    def backward(self, inputs, dvalues):
        # reshape input gradient
        self.dvalues = dvalues.reshape(self.size_batch, self.H, self.W, self.D)

        # initialize output gradients
        self.dinputs = []
        self.dkernels = np.zeros_like(self.kernels)

        for idx, x in enumerate(inputs):
            # initialize gradient tensors
            dinput = np.zeros_like(x)

            for h in range(self.H):
                for w in range(self.W):
                    for d in range(self.D):
                        # extract current sample
                        sample = x[h:h + self.h_k, w:w + self.w_k, :]

                        # calculate the gradient of the loss
                        sample_out = self.dvalues[idx, h, w, d]

                        # calculate output gradients
                        dinput[h:h + self.h_k, w:w + self.w_k, :] += \
                            self.kernels[:, :, d, np.newaxis] * sample_out

                        self.dkernels[:, :, d, np.newaxis] += sample_out * sample

            # store batch-wise gradients
            self.dinputs.append(dinput)

        #
        self.dinputs = np.array(self.dinputs)


# basic convolutional neural network optimizer
class OptimizerCNN:

    # initialize optimizer: genearlly LR @ 1e-3, decay @ 1e-4
    def __init__(self, learning_rate=0.001, decay=1e-4):
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
        # update kernels and biases using mean gradients
        layer.kernels -= self.current_lr * np.mean(layer.dkernels, axis=0)
        layer.biases -= self.current_lr * np.mean(layer.dvalues, axis=0)

    # update internal loop
    def post_update(self):
        self.iterations += 1