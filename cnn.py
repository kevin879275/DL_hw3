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

print("img = \n", img)
if padding > 0:
    img = np.pad(array=img, pad_width=((0, 0), (0, 0), (padding, padding),
                                       (padding, padding)), mode='constant', constant_values=0)
weights = np.random.randint(
    4, size=(out_channels, in_channels, kernel_size[0], kernel_size[1]))
biases = np.random.randint(10, size=out_channels)

output_shape_H = int(
    ((img_H - kernel_size[0] + 2*padding) / strides) + 1)
output_shape_W = int(
    ((img_W - kernel_size[1] + 2*padding) / strides) + 1)
output = np.zeros((BATCH_SIZE, out_channels, output_shape_H, output_shape_W))

print("padding_img = \n", img)
print("weights = \n", weights)

for channel in range(out_channels):
    for H in range(output_shape_H):
        for W in range(output_shape_W):
            dot = np.multiply(img[:, :, H * strides: H*strides + kernel_size[0],
                                  W*strides:W*strides+kernel_size[1]], weights[channel])
            s = np.sum(dot, axis=(1, 2, 3))
            s = s + biases[channel]
            output[:, channel, H, W] = s


print("output : \n", output)
