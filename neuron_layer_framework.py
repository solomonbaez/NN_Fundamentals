# Conceptual simulation of a three-neuron NN layer

# inputs fed into the neuron may be neuronal outputs or external data
inputs = [1, 2, 3, 2.5]

# weight is unique to each neuron
# linearly mutates the input values using a constant
weights1 = [0.4, 0.2, -0.6, 1.0]
weights2 = [0.2, 0.10, 0.6, -1.0]
weights3 = [-0.48, 2, 0.53, 0.67]

# bias is distinct and unique to each neuron
# offsets the input values using a constant
bias1 = 2
bias2 = 3
bias3 = 0.5

output1 = output2 = output3 = 0

# simulated neuronal output based on previous parameters
# generally, output = input*weight + bias
for i in range(len(inputs)):
    output1 += inputs[i] * weights1[i] + bias1
    output2 += inputs[i] * weights2[i] + bias2
    output3 += inputs[i] * weights3[i] + bias3

# return neuronal outputs
print(output1, output2, output3)
