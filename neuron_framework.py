
#GENERALLY neuronal outputs will become the
#inputs of the following layer

#inputs, weights, and biases are unique
inputs = [1]*3
weights = [2]*3
bias = 3
output = 0

for i in range(len(inputs)):
    output += inputs[i]*weights[i] + bias

print(output)
