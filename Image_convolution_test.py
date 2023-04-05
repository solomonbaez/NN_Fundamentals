import numpy as np
from keras.datasets import mnist

# import data
# X values correspond to a uint8 np.array of grayscale data
# X.shape = (6000, 28, 28)
# y values correspond to a uint8 np.array of digit labels (0-9)
# y.shape = (6000,)
(X, y), (X_valid, y_valid) = mnist.load_data()

# truncate features
X = X[:10]

# set the kernel dimensions
kernel_shape = (3, 3)
kernel = np.random.randn(*kernel_shape)

# define input dimensions
x = X[0]
x_h = x.shape[0]
x_w = x.shape[1]

# define kernel dimensions
k_h = kernel.shape[0]
k_w = kernel.shape[1]

# define zero-padding
h, w = k_h // 2, k_w // 2

# pre-set output tensor
out = np.zeros(X.shape)

for x in X:
    for o in out:
        # correlate the input with the kernel
        for i in range (h, x_h - h):
            for j  in range(w, x_w - w):
                # temporary sum variable
                tmp = 0

                # find correlation at output matrix position [i][j]
                for m in range (k_h):
                    for n in range(k_w):
                        # find dot products along kernel heights and widths
                        tmp += kernel[m][n]*x[i-h+m][j-w+n]

                # set correlation
                o[i][j] = tmp

# # if desired, view input and output tensor samples
# import matplotlib.pyplot as plt
# sample = plt.imshow(X[0], cmap='gray', vmin=0, vmax=255)
# output = plt.imshow(out[0], cmap='gray', vmin=0, vmax=255)
# plt.show()


# verify shape is maintained
print(X.shape, out.shape)



