from optimizers import *
from layers_activators import *

import numpy as np

# import and initialize training data from nnfs
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

# create dataset
X, y = spiral_data(samples=1000, classes=3)

# primary layer, 2 input features, 64 outputs
layer1 = DeepLayer(2, 64, l2_w=5e-4, l2_b=5e-4)

# primary layer activation using the ReLU method
activation1 = ReLU()

# secondary layer, 64 input features, 3 classes
layer2 = DeepLayer(64, 3)

# SoftMax classifier's combined activation and CCE loss
cbp = CombinedBP()

# optimizer initialization
optimizer = OptimizerAdaM(learning_rate=0.05, decay=5e-7)

# train the model
for epoch in range(10001):
    # forward passes on the first layer/activation pair
    layer1.forward(X)
    activation1.forward(layer1.output)

    # forward pass on layer two
    layer2.forward(activation1.output)

    # activation/loss on layer two
    data_loss = cbp.forward(layer2.output, y)

    regularization_loss = cbp.loss.regularization(layer1) \
                          + cbp.loss.regularization(layer2)

    loss = data_loss + regularization_loss

    # store activation2 outputs along axis 1
    predictions = np.argmax(cbp.output, axis=1)
    # discretize one-hot encoded targets
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)

    # calculate accuracy from activation2 and targets
    accuracy = np.mean(predictions==y)
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
    # combined SoftMax and CCE backwards step implicit to the cbp object
    cbp.backward(cbp.output, y)
    # reverse sequential utilization of layer/activation gradients
    layer2.backward(cbp.dinputs)
    activation1.backward(layer2.dinputs)
    layer1.backward(activation1.dinputs)

    # decay the learning rate
    optimizer.lr_decay(epoch)
    # update weights and biases in the model layers
    optimizer.update(layer1)
    optimizer.update(layer2)

# Validate the model

X_test, y_test = spiral_data(samples=100, classes=3)

# pass the testing data forward through both model layers
layer1.forward(X_test)
activation1.forward(layer1.output)
layer2.forward(activation1.output)
loss_valid = cbp.forward(layer2.output, y_test)

# calculate accuracy
predictions_valid = np.argmax(cbp.output, axis=1)
if len(y_test.shape) == 2:
    y_test = np.argmax(y_test, axis=1)
accuracy_valid = np.mean(predictions_valid == y_test)

# report validation statistics
print(f"validation, accuracy: {accuracy_valid:.3f}, loss: {loss_valid:.3f}")
