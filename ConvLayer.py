import numpy as np

# Convolutinal Layer: single kernel functionality
class ConvLayer:
    def __init__(self, inputs, shape_in, shape_k):
        # store inputs
        self.X_inputs = inputs
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

    def forward(self, inputs, training=False):
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

    def backward(self, dvalues):
        # reshape input gradient
        self.dvalues = dvalues.reshape(self.size_batch, self.H, self.W, self.D)

        # initialize output gradients
        self.dinputs = []
        self.dkernels = np.zeros_like(self.kernels)

        for idx, x in enumerate(self.X_inputs):
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

# Adaptive Momentum optimizer: customized for CNN
class CNNAdaM:

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
    def pre_update(self):
        if self.decay:
            self.current_lr = self.lr / (1 + self.decay * self.iterations)

    # update layer parameters using optimizer settings
    def update(self, layer):

        # if absent, create cache arrays within the layer
        layer.momentum_b = np.zeros_like(layer.biases)
        layer.cache_b = np.zeros_like(layer.biases)

        if not hasattr(layer, 'cache_w') and hasattr(layer, "weights"):
            layer.momentum_w = np.zeros_like(layer.weights)
            layer.cache_w = np.zeros_like(layer.weights)

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

        if hasattr(layer, "kernels"):
            layer.momentum_k = np.zeros_like(layer.kernels)
            layer.cache_k = np.zeros_like(layer.kernels)

            layer.momentum_k = self.beta1 * layer.momentum_k + (1 - self.beta1) \
                               * np.mean(layer.dkernels, axis=0)

            layer.momentum_b = self.beta1 * layer.momentum_b + (1 - self.beta1) \
                               * np.mean(layer.dvalues, axis=0)

            momentum_k_corrected = layer.momentum_k / \
                                   (1 - self.beta1 ** (self.iterations + 1))

            momentum_b_corrected = layer.momentum_b / \
                                   (1 - self.beta1 ** (self.iterations + 1))

            layer.cache_k = self.beta2 * layer.cache_k + (1 - self.beta2) \
                            * np.mean(layer.dkernels, axis=0) ** 2

            layer.cache_b = self.beta2 * layer.cache_b + (1 -self.beta2) \
                            * np.mean(layer.dvalues, axis=0) ** 2

            cache_k_corrected = layer.cache_k / \
                                (1 - self.beta2 ** (self.iterations + 1))
            cache_k_corrected = np.mean(cache_k_corrected, axis=0)

            cache_b_corrected = layer.cache_b / \
                                (1 - self.beta2 ** (self.iterations + 1))
            cache_b_corrected = np.mean(cache_b_corrected, axis=0)

            # The cache's means are taken to modify the momentum
            layer.kernels += -self.current_lr * momentum_k_corrected / \
                             (np.sqrt(cache_k_corrected) + self.epsilon)
            layer.biases += -self.current_lr * momentum_b_corrected / \
                            (np.sqrt(cache_b_corrected) + self.epsilon)

    def post_update(self):
        self.iterations += 1