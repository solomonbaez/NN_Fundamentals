import numpy as np
from keras.datasets import mnist


# Dense Layer: 2D matrix processing
class DenseLayer:
    # weight initializer utilized in weight distribution modification
    # used when model will not learn in accordance with learning rate adjustments
    def __init__(self, n_in, n_neurons, init_w=0.01):
        # initialize input size
        self.weights = init_w * np.random.randn(n_in, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs, training=False):
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


# Convolutinal Layer
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


# Rectified Linear Unit activation function
class ReLU:
    # calculate predictions for model outputs
    def predict(self, outputs):
        return outputs

    def forward(self, inputs, training=True):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        # copy the input gradient values and structure
        self.dinputs = dvalues.copy()
        # produce a zero gradient where input values were invalid
        self.dinputs[self.inputs <= 0] = 0


# SoftMax activation function
class SoftMax:
    # calculate predictions for model outputs
    def predict(self, outputs):
        return np.argmax(outputs, axis=1)

    def forward(self, inputs, training=False):
        self.inputs = inputs

        # exponential function application and normalization by the maximum input
        exponentials = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # normalization per softmax methodology
        self.output = exponentials/np.sum(exponentials, axis=1, keepdims=True)

    def backward(self, dvalues):
        # uninitialized array
        self.dinputs = np.empty_like(dvalues)

        for i, (out, dvalue) in enumerate(zip(self.output, dvalues)):
            # flatten the output
            out = out.reshape(-1, 1)
            # calculate jacobian matrix
            jacobian = np.diagflat(out) - np.dot(out, out.T)
            # calculate sample-wise gradient and store
            self.dinputs[i] = np.dot(jacobian, dvalue)


# common Loss class
class Loss:
    # calculate data loss and return losses
    def calculate(self, inputs, targets):
        sample_loss = self.forward(inputs, targets)

        data_loss = np.mean(sample_loss)

        return data_loss


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

    def backward(self, dvalues, targets):
        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(targets.shape) == 1:
            targets = np.eye(labels)[targets]

        self.dinputs = -targets / dvalues
        self.dinputs = self.dinputs/ samples


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


# Adaptive Momentum Optimizer
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


# import data
(X, y), (X_valid, y_valid) = mnist.load_data()

# truncate and normalize features
X, y = X[:1000], y[:1000]

X = X.reshape(len(X), 28, 28, 1)

X = X.astype("float32") / 255

# define kernel shape
kernel = (3, 3, 1)

# define primary convolutional layer/activation pair: 3x3 kernel with 1 input channel
clayer = ConvLayer(shape_in=X.shape, shape_k=kernel)
reluCNN = ReLU()

# define secondary layer/activation pair
dlayer = DenseLayer(clayer.H * clayer.W * clayer.D, len(X))
reluD = ReLU()

# define final layer/activation pair: 10 outputs in accordance with ground truth labels (1-10)
flayer = DenseLayer(len(X), 10)
softmax = SoftMax()

# define loss
cce = LossCCE()

# set optimizers
optimizerCNN = OptimizerCNN(learning_rate=1e-3, decay=2e-3)
optimizerD = OptimizerAdaM(learning_rate=1e-3, decay=2e-3)

# train the model
for epoch in range(1001):
    # forward pass on first layer/activator pair
    clayer.forward(X)
    reluCNN.forward(clayer.output)

    # forward pass on second layer/activator pair
    dlayer.forward(reluCNN.output)
    reluD.forward(dlayer.output)

    # forward pass on final layer/activation pair
    flayer.forward(reluD.output)
    softmax.forward(flayer.output)

    # calculate loss
    loss = cce.calculate(softmax.output, y)

    # report strongest predictions and calculate accuracy
    predictions = np.argmax(softmax.output, axis=1)
    accuracy = np.mean(predictions == y)

    # report model performance
    if not epoch % 10:
        print(f"epoch: {epoch}, " +
              f"accuracy: {accuracy:.3f}, " +
              f"loss: {loss:.3f}, " +
              f"lr: {optimizerCNN.current_lr}")

    # backpropogate the model
    cce.backward(softmax.output, y)
    softmax.backward(cce.dinputs)
    flayer.backward(softmax.dinputs)
    reluD.backward(flayer.dinputs)
    dlayer.backward(reluD.dinputs)
    reluCNN.backward(dlayer.dinputs)
    clayer.backward(X, reluCNN.dinputs)

    # update layers
    optimizerCNN.pre_update()
    optimizerD.pre_update()
    optimizerCNN.update(clayer)
    optimizerD.update(dlayer)
    optimizerD.update(flayer)
    optimizerCNN.post_update()
    optimizerD.post_update()
