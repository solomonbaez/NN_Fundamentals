import numpy as np


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
    def pre_update(self):
        if self.decay:
            self.current_lr = self.lr / (1 + self.decay * self.iterations)

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

    def post_update(self):
        self.iterations += 1
