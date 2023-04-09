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


# Convolutinal Layer: single kernel functionality
class ConvLayer:
    def __init__(self, shape_in, shape_k):
        # initialize the kernel(s)
        self.kernels = np.random.randn(*shape_k)

        # initialize dimensions, kernel dimensions must be accessable
        _, self.h_in, self.w_in, self.d_in = shape_in
        self.h_k, self.w_k, self.d_k = self.kernels.shape

        # initialize the output tensor
        self.shape_out = (self.h_in - self.h_k + 1,
                          self.w_in - self.w_k + 1,
                          self.d_k)

        self.biases = np.random.randn(*self.shape_out)

    def forward(self, inputs):
        # define convolution dimensions
        H, W, D = self.shape_out

        # initialize output storage tensor
        self.output = []

        # loop input features through output tensor dimensions
        for idx, x in enumerate(inputs):
            # set storage tensor
            self.out = np.copy(self.biases)
            for h in range(H):
                for w in range(W):
                    for d in range(D):
                        # extract the current sample
                        sample = x[h:h+self.h_k, w:w+self.w_k]

                        # correlate the sample with the kernel
                        correlation = sample * self. kernels[:, :, d]

                        self.out[h, w, d] += np.sum(correlation)
            # store batch-wise output
            self.output.append(self.out)
        self.output = np.array(self.output)

        # restructure the output
        self.output = self.output.reshape((100, 26*26))

    def backward(self, inputs, dvalues):
        self.dvalues = dvalues
        self.dinputs = []
        self.dkernels = []

        for idx, x in enumerate(inputs):
            # print(x.shape, "x shape")
            # initialize gradient tensors
            dinput = np.zeros_like(inputs[0])
            dkernel = np.zeros_like(self.kernels)
            # print(dinput.shape, "dinput shape")
            # print(dkernel.shape, "dkernel shape")
                # print(dvalue.shape, "dvalue shape")
            for h in range(self.h_in - self.h_k + 1):
                for w in range(self.w_in - self.w_k + 1):
                    for d in range(self.d_k):
                        # extract current sample
                        sample = x[h:h + self.h_k, w:w + self.w_k, :]
                        # print(sample.shape, "sample shape")
                        # calculate the gradient of the loss
                        sample_out = dvalues[idx, h, w, d]
                        # print(sample_out, "dvalue sample")
                        # calculate output gradients
                        dinput[h:h + self.h_k, w:w + self.w_k, :] += \
                            self.kernels[:, :, :] * sample_out
                        dkernel[:, :, :] += sample * sample_out
            # store batch-wise gradients
            self.dinputs.append(dinput)
            self.dkernels.append(dkernel)
        self.dinputs = np.array(self.dinputs)
        self.dkernels = np.array(self.dkernels)


# Sigmoid Activation
class Sigmoid:
    # calculate predictions for model outputs
    def predict(self, outputs):
        return (outputs > 0.5) * 1

    def forward(self, inputs):
        self.inputs = inputs

        # Sigmoid activation
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues):
        # Partial derivative of the sigmoid activation function
        # s' = s(1 - s)
        self.dinputs = dvalues * (1 - self.output) * self.output


# common Loss class
class Loss:
    # calculate data loss and return losses
    def calculate(self, inputs, targets):
        sample_loss = self.forward(inputs, targets)

        data_loss = np.mean(sample_loss)

        return data_loss


# Binary Cross-Entropy Loss
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

        # return data loss
        return np.mean(sample_losses)

    def backward(self, dvalues, targets):
        # prevent /0 process death without impacting the mean
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)

        # calculate and normalize the gradient
        self.dinputs = -(targets / clipped_dvalues -
                         (1 - targets) / (1 - clipped_dvalues))
        self.dinputs = self.dinputs / len(dvalues)


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

# import data
# X values correspond to a uint8 np.array of grayscale data
# X.shape = (6000, 28, 28)
# y values correspond to a uint8 np.array of digit labels (0-9)
# y.shape = (6000,)
(X, y), (X_valid, y_valid) = mnist.load_data()

# truncate and normalize features
X, y = X[:100], y[:100]

X = X.reshape(len(X), 28, 28, 1)

X = X.astype("float32") / 255

# define primary convolutional layer/activation pair: single 3x3 kernel
clayer = ConvLayer(shape_in=X.shape, shape_k=(3,3,1))
s1 = Sigmoid()

# define secondary layer/activation pair
dlayer = DenseLayer(676, 100)
s2 = Sigmoid()

# binary loss
bloss = BinaryCE()

# optimizer
optimizer = OptimizerCNN()

for epoch in range(1001):
    # forward pass on first layer/activator pair
    clayer.forward(X)
    s1.forward(clayer.output)

    # forward pass on second layer/activator pair
    dlayer.forward(s1.output)
    s2.forward(dlayer.output)

    # calculate loss and accuracy: CURRENT VALUES INACCinaURATE
    loss = bloss.calculate(s2.output, y)

    predictions = (s2.output > 0.5) * 1
    accuracy = np.mean(predictions == y)

    # report model performance
    if not epoch % 100:
        print(f"epoch: {epoch}, " +
              f"accuracy: {accuracy:.3f}, " +
              f"loss: {loss:.3f}, " +
              f"lr: {optimizer.current_lr}")

    # backpropogate the model
    bloss.backward(s2.output, y)
    s2.backward(bloss.dinputs)
    dlayer.backward(s2.dinputs)
    s1.backward(dlayer.dinputs)
    # reshape the primary Sigmoid gradient
    final_d = s1.dinputs.reshape(100, 26, 26, 1)
    clayer.backward(X, final_d)

    # update layers
    optimizer.pre_update()
    optimizer.update(clayer)
    optimizer.post_update()