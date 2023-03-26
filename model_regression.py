from optimizers import *
from layers_activators import *

import numpy as np

# import and initialize training data from nnfs
import nnfs
from nnfs.datasets import sine_data

nnfs.init()

# create sine wave dataset
X, y = sine_data()

# primary layer, 1 input feature, 64 outputs
# note weight initializer adjustment to account for model inability to fit
layer1 = DeepLayer(1, 64, init_w=0.1)

# primary layer activation using the ReLU method
activation1 = ReLU()

# secondary hidden layer, 64 input features, 64 outputs
# model will not learn without multiple hidden layers due to ReLU functionality
# ReLU allows mapping of nonlinear behavior over successive layer iterations
layer2 = DeepLayer(64, 64, init_w=0.1)

# secondary layer activation using the ReLU method
activation2 = ReLU()

# tertiary layer, 64 input features, 1 output
layer3 = DeepLayer(64, 1, init_w=0.1)

# tertiary layer activation using the Linear method
activation3 = Linear()

# loss function
mse = MSE()

# optimizer initialization
# 0.005 is a near perfect learning rate for this data
# at 0.001 and 0.01 the data could not be fit prior to weight initializer adjustment
optimizer = OptimizerAdaM(learning_rate=0.005, decay=1e-3)

# precision used in calculating accuracy
# regression does not allow for accuracy in the same way as classification
# simulated accuracy requires checking values against their ground truth equivalents
# the difference between the two values is then compared against the precision metric
precision = np.std(y) / 250

# train the model
for epoch in range(10001):
    # forward pass on the first two layer/activation pairs
    layer1.forward(X)
    activation1.forward(layer1.output)

    # hidden layer
    layer2.forward(activation1.output)
    activation2.forward(layer2.output)

    # forward pass on the third layer/activation pair
    layer3.forward(activation2.output)
    activation3.forward(layer3.output)

    # calculate data loss
    data_loss = mse.calculate(activation3.output, y)

    # calculate regularization loss
    regularization_loss = mse.regularization(layer1) + \
                          mse.regularization(layer2) + \
                          mse.regularization(layer3)

    # calculate overall loss
    loss = data_loss + regularization_loss

    # calculate and report simulated accuracy
    predictions = activation3.output
    accuracy = np.mean(np.absolute(predictions - y) < precision)

    if not epoch % 100:
        print(f"epoch: {epoch}, " +
              f"accuracy: {accuracy:.3f}, " +
              f"loss: {loss:.3f}, " +
              f"data_loss: {data_loss:.3f}, " +
              f"regularization_loss: {regularization_loss:.3f}, " +
              f"lr: {optimizer.current_lr:.3f}")

    # backpropogation
    mse.backward(activation3.output, y)
    activation3.backward(mse.dinputs)
    layer3.backward(activation3.dinputs)
    activation2.backward(layer3.dinputs)
    layer2.backward(activation2.dinputs)
    activation1.backward(layer2.dinputs)
    layer1.backward(activation1.dinputs)

    # decay the learning rate
    optimizer.pre_update()

    # update weights and biases
    optimizer.update(layer1)
    optimizer.update(layer2)
    optimizer.update(layer3)

    # update internal loop
    optimizer.post_update()

# validate and plot model performance
import matplotlib.pyplot as plt

X_test, y_test = sine_data()

layer1.forward(X_test)
activation1.forward(layer1.output)
layer2.forward(activation1.output)
activation2.forward(layer2.output)
layer3.forward(activation2.output)
activation3.forward(layer3.output)

plt.plot(X_test, y_test)
plt.plot(X_test, activation3.output)

plt.show()