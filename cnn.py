import numpy as np

in_channels = 3
out_channels = 2

BATCH_SIZE = 2
padding = 1
strides = 1

img_H = 5
img_W = 5
kernel_size = (3, 3)

img = np.random.randint(4, size=(BATCH_SIZE, in_channels, img_H, img_W))
grad_output = np.zeros(img.shape)
print("img = \n", img)
if padding > 0:
    img = np.pad(array=img, pad_width=((0, 0), (0, 0), (padding, padding),
                                       (padding, padding)), mode='constant', constant_values=0)
weights = np.random.randint(4,
                            size=(out_channels, in_channels, kernel_size[0], kernel_size[1]))
grad_weights = np.zeros((out_channels, in_channels,
                         kernel_size[0], kernel_size[1]))
biases = np.random.randint(10, size=out_channels)

grad_input_shape_H = int(
    ((img_H - kernel_size[0] + 2*padding) / strides) + 1)
grad_input_shape_W = int(
    ((img_W - kernel_size[1] + 2*padding) / strides) + 1)
grad_input = np.random.randint(
    5, size=(BATCH_SIZE, out_channels, grad_input_shape_H, grad_input_shape_W))
if padding > 0:
    grad_input_padding = np.pad(array=grad_input, pad_width=((0, 0), (0, 0), (padding, padding),
                                                             (padding, padding)), mode='constant', constant_values=0)

print("padding_img = \n", img)
print("grad_inputs = \n", grad_input)

for out_channel in range(grad_weights.shape[0]):
    for H in range(grad_weights.shape[2]):
        for W in range(grad_weights.shape[3]):
            x = img[:, :, H*strides:H * strides+kernel_size[0],
                    W*strides:W*strides+kernel_size[1]]
            y = grad_input[:, out_channel, H, W]
            for i in range(x.shape[0]):
                grad_weights[out_channel, :, :, :] += x[i]*y[i]
            # grad_weights[:, channel, :, :] += *

grad_biases = np.mean(grad_input, axis=0)

grad_weights = grad_weights / BATCH_SIZE

weights_inverse = np.rot90(weights, axes=(2, 3))


for in_channel in range(in_channels):
    for H in range(grad_output.shape[2]):
        for W in range(grad_output.shape[3]):
            break
            # dot = np.multiply(grad_input[:, channel,H*strides:H*strides+kernel_size[0],W*strides:W*strides + kernel_size[1]],weights_inverse[out_channels,])
print("grad_weights : \n", grad_weights)
