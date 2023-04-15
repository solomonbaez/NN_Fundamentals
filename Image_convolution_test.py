import numpy as np
from keras.datasets import mnist
from layers import DenseLayer
from ConvLayer import *
from activators import ReLU, SoftMax, SoftMaxCCE
from loss import LossCCE
from optimizers import OptimizerAdaM


# import data
(X, y), (X_valid, y_valid) = mnist.load_data()

# truncate and normalize features
X, y = X[:100], y[:100]
X_valid, y_valid = X_valid[:100], y_valid[:100]

X, X_valid = X.reshape(len(X), 28, 28, 1), X_valid.reshape(len(X_valid), 28, 28, 1)
X, X_valid = X.astype("float32") / 255, X_valid.astype("float32") / 255

# define kernel shape
kernel = (3, 3, 1)

# define primary convolutional layer/activation pair: 3x3 kernel with 1 input channel
Conv1 = ConvLayer(shape_in=X.shape, shape_k=kernel)
reluC = ReLU()

# define secondary layer/activation pair
Dense1 = DenseLayer(Conv1.H * Conv1.W * Conv1.D, len(X))
reluD = ReLU()

# define final layer/activation pair: 10 outputs in accordance with ground truth labels (1-10)
Dense2 = DenseLayer(len(X), 10)
softmax = SoftMax()

# define loss
cce = LossCCE()
cce_backward = SoftMaxCCE()

# set optimizers
optimizerC = OptimizerCNN(learning_rate=1e-3, decay=2e-3)
optimizerD = OptimizerAdaM(learning_rate=1e-3, decay=2e-3)


# train the model
for epoch in range(1001):
    # forward pass on first layer/activator pair
    Conv1.forward(X)
    reluC.forward(Conv1.output)

    # forward pass on second layer/activator pair
    Dense1.forward(reluC.output)
    reluD.forward(Dense1.output)

    # forward pass on final layer/activation pair
    Dense2.forward(reluD.output)
    softmax.forward(Dense2.output, training=True)

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
              f"lr: {optimizerC.current_lr}")

    # backpropogate the model
    cce_backward.backward(softmax.output, y)
    Dense2.backward(cce_backward.dinputs)
    reluD.backward(Dense2.dinputs)
    Dense1.backward(reluD.dinputs)
    reluC.backward(Dense1.dinputs)
    Conv1.backward(X, reluC.dinputs)

    # update layers
    optimizerC.pre_update()
    optimizerD.pre_update()
    optimizerC.update(Conv1)
    optimizerD.update(Dense1)
    optimizerD.update(Dense2)
    optimizerC.post_update()
    optimizerD.post_update()


# model validation
Conv1.forward(X_valid)
reluC.forward(Conv1.output)

Dense1.forward(reluC.output)
reluD.forward(Dense1.output)

Dense2.forward(reluD.output)
softmax.forward(Dense2.output, training=False)

# calculate validation loss
loss_valid = cce.calculate(softmax.output, y_valid)

# calculate strongest validation predictions and calculate validation accuracy
predictions_valid = np.argmax(softmax.output, axis=1)
accuracy_valid = np.mean(predictions_valid == y_valid)

# report validation performace
print(f"validation, accuracy: {accuracy_valid:.3f}, loss: {loss_valid:.3f}")
