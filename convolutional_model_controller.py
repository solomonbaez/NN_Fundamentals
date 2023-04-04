from CNNLayer import *
from layers import DenseLayer
from activators import Sigmoid, ReLU
from loss import BinaryCE
from optimizers import OptimizerAdaM
from keras.datasets import mnist


# generate dataset
(X_train, y_train), (X_valid, y_valid) = mnist.load_data()
X_train, y_train = process(X_train, y_train, 10)
X_valid, y_valid = process(X_valid, y_valid, 10)

model = [
    # set CNNLayer, define kernel parameters
    CNNLayer(shape=X_train[0].shape, size_k=3, depth_k=5),
    Sigmoid(),
    ReshapeLayer(shape_in=(5,26,26) , shape_out=(5*26*26, 1)),
    DenseLayer(1, 5*26*26),
    ReLU(),
    DenseLayer(5*26*26, 2),
    Sigmoid(),
]

binary_loss = BinaryCE()
optimizer = OptimizerCNN(learning_rate=1e-1, decay=1e-3)
optimizerD = OptimizerAdaM(learning_rate=1e-1, decay=1e-3)

# # train, currently broken
# for epoch in range(1001):
#     print(epoch)
#     loss, accuracy = 0, 0
#
#     for X, y in zip(X_train, y_train):
#         output = X
#         for layer in model:
#             layer.forward(output)
#             output = layer.output
#
#         loss += (binary_loss.calculate(output.T, y))/len(X_train)
#
#         predictions = (output.T > 0.5) * 1
#         # calculate accuracy
#         accuracy += (np.mean(predictions == y))/len(X_train)
#
#     binary_loss.backward(output.T, y)
#     dinputs = binary_loss.dinputs.T
#
#     for i, layer in enumerate(reversed(model)):
#         print(i)
#         print(dinputs.shape)
#
#         if i == 2:
#             dinputs = dinputs.T
#             print(dinputs.shape)
#             print(layer.inputs.shape)
#
#         layer.backward(dinputs)
#         dvalues = layer.dinputs
#
#         if i == 2:
#             dinputs = dinputs.T
#
#     # report epoch performance
#     # generally test loss == training loss on a high performing model
#     # 10% difference in loss is indicative of significant issues
#     if not epoch % 5:
#         print(f"epoch: {epoch}, " +
#               f"accuracy: {accuracy:.3f}, " +
#               f"loss: {loss:.3f}, " +
#               f"lr: {optimizer.current_lr}")
#
#     # # update layers
#     # optimizer.pre_update()
#     # optimizer.update(model[0])
#     # optimizer.post_update()
#     #
#     # optimizerD.pre_update()
#     # optimizerD.update(model[3])
#     # optimizerD.update(model[5])
#     # optimizerD.post_update()