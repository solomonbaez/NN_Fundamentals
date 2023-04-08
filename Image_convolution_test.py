import numpy as np
from keras.datasets import mnist

# import data
# X values correspond to a uint8 np.array of grayscale data
# X.shape = (6000, 28, 28)
# y values correspond to a uint8 np.array of digit labels (0-9)
# y.shape = (6000,)
(X, y), (X_valid, y_valid) = mnist.load_data()

# truncate features
X = X[:100]

X = X.reshape(len(X), 28, 28, 1)

# set the kernel dimensions
kernel_shape = (3, 3, 1)
kernel = np.random.randn(*kernel_shape)

# define input dimensions
_, x_h, x_w, x_d = X.shape
k_h, k_w, k_d = kernel.shape

# define output dimensions
out_h = x_h - k_h + 1
out_w = x_w - k_w + 1
out_d = k_d

# pre-set output tensor
out = np.zeros((out_h, out_w, out_d))
out_array = []

for x in X:
    for h in range(out_h):
        for w in range(out_w):
            for d in range(out_d):
                # extract image patch
                sample = x[h:h+k_h, w:w+k_w, :]

                # correlate the sample and kernel
                corr = sample * kernel[:, :, d]

                # sum the results
                out[h, w, d] = np.sum(corr)
    out_array.append(out)

out_array = np.array(out_array)

# # if desired, view input and output tensor samples
# import matplotlib.pyplot as plt
# sample = plt.imshow(X[0], cmap='gray', vmin=0, vmax=255)
# output = plt.imshow(out[0], cmap='gray', vmin=0, vmax=255)
# plt.show()


# verify shape is maintained
print(X.shape, out_array.shape)



