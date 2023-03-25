from optimizers import *
from layers_activators import *

import numpy as np

# import and initialize training data from nnfs
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

# create dataset, set 2 classes for binary logistic regression
X, y = spiral_data(samples=100, classes=2)

# reshape labels to be a nested list
# inner list contains 0 or 1
y = y.reshape(-1, 1)

# primary layer, 2 input features, 64 outputs
layer1 = DeepLayer(2, 64, l2_w=5e-4, l2_b=5e-4)

# primary layer activation using the ReLU method
activation1 = ReLU()

# secondary layer, 64 input features, 1 class
layer2 = DeepLayer(64, 1)

# secondary layer activation using the Sigmoid method
activation2 = Sigmoid()

# loss function
binary_loss = BinaryCE()

# optimizer initialization
optimizer = OptimizerAdaM(decay=5e-7)

# train the model
for epoch in range(10001):
    # forward passes on the first layer/activation pair
    layer1.forward(X)
    activation1.forward(layer1.output)

    # forward pass on layer two
    layer2.forward(activation1.output)

    # sigmoid activation
    activation2.forward(layer2.output)

    # activation/loss on layer two
    data_loss = binary_loss.calculate(activation2.output, y)

    # regularization loss
    regularization_loss = binary_loss.regularization(layer1) + \
                          binary_loss.regularization(layer2)

    # overall loss
    loss = data_loss + regularization_loss

    # calculate predictions from a binary mask
    # mutate into a binary array
    predictions = (activation2.output > 0.5) * 1

    # calculate accuracy
    accuracy = np.mean(predictions == y)

    # report epoch performance
    # generally test loss == training loss on a high performing model
    # 10% difference in loss is indicative of significant issues
    if not epoch % 100:
        print(f"epoch: {epoch}, " +
              f"accuracy: {accuracy:.3f}, " +
              f"loss: {loss:.3f}, " +
              f"data_loss: {data_loss:3f}, " +
              f"regulization_loss: {regularization_loss:.3f}, " +
              f"lr: {optimizer.current_lr}")

    # backpropogation, starting at the most recent forward pass
    # reverse sequential utilization of layer/activation gradients
    binary_loss.backward(activation2.output, y)
    activation2.backward(binary_loss.dinputs)
    layer2.backward(activation2.dinputs)
    activation1.backward(layer2.dinputs)
    layer1.backward(activation1.dinputs)

    # decay the learning rate
    optimizer.pre_update()
    # update weights and biases in the model layers
    optimizer.update(layer1)
    optimizer.update(layer2)
    optimizer.post_update()

# Validate the model

X_test, y_test = spiral_data(samples=100, classes=2)
# reshape labels
y_test = y_test.reshape(-1, 1)

# pass the testing data forward through both model layers
layer1.forward(X_test)
activation1.forward(layer1.output)
layer2.forward(activation1.output)
activation2.forward(layer2.output)

loss_valid = binary_loss.calculate(activation2.output, y_test)

# calculate accuracy
predictions_valid = (activation2.output > 0.5) * 1
accuracy_valid = np.mean(predictions_valid == y_test)

# report validation statistics
# utilizing dropout the validation set performs better than the training set
print(f"validation, accuracy: {accuracy_valid:.3f}, loss: {loss_valid:.3f}")
