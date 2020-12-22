import numpy as np


class FC():
    def __init__(self, input, output, lr=0.1):
        self.input = input
        self.output = output
        self.lr = lr
        self.weight = np.random.normal(loc=0.0, scale=np.sqrt(2/(input+output)),
                                       size=(self.input, self.output))
        # self.weight = np.random.normal(loc=0.0, scale=1,
        #                                size=(self.input, self.output))
        self.biases = np.zeros(output)

    def forward(self, input):
        return np.dot(input, self.weight) + self.biases

    def backward(self, input, grad_input):
        grad_output = np.dot(grad_input, (self.weight).T)
        # (d z / d w) * ( d C / d z) = ( d C / d w)
        grad_weights = np.dot(input.T, grad_input)
        grad_biases = grad_input.mean(axis=0)
        self.weight = self.weight - self.lr * grad_weights
        self.biases = self.biases - self.lr * grad_biases
        return grad_output


class ReLU():
    def __init__(self):
        pass

    def forward(self, input):
        return np.maximum(input, 0)

    def backward(self, input, grad_input):
        relu_grad = input > 0
        return grad_input*relu_grad


class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, strides=1, padding=1, lr=0.01):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.lr = lr
        # self.weights = self.weights = np.random.normal(loc=0.0, scale=1,
        #                                                size=(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        self.weights = self.weights = np.random.randn(
            out_channels, in_channels, kernel_size[0], kernel_size[1]) / (kernel_size[0] * kernel_size[1])
        self.biases = np.zeros(out_channels)

    def forward(self, input):
        # print("original_img = \n", input)
        img_H = input.shape[2]
        img_W = input.shape[3]

        if self.padding > 0:
            input_padding = np.pad(array=input, pad_width=(
                (0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant', constant_values=0)

        output_shape_H = int(
            ((img_H - self.kernel_size[0] + 2*self.padding) / self.strides) + 1)
        output_shape_W = int(
            ((img_W - self.kernel_size[1] + 2*self.padding) / self.strides) + 1)
        output = np.zeros(
            (input.shape[0], self.out_channels, output_shape_H, output_shape_W))

        # print("padding_img = \n", input_padding)
        # print("weights = \n", self.weights)

        for channel in range(self.out_channels):
            for H in range(output_shape_H):
                for W in range(output_shape_W):
                    dot = np.multiply(input_padding[:, :, H * self.strides: H*self.strides + self.kernel_size[0],
                                                    W*self.strides:W*self.strides+self.kernel_size[1]], self.weights[channel])
                    s = np.sum(dot, axis=(1, 2, 3))
                    s = s + self.biases[channel]
                    output[:, channel, H, W] = s

        #print("output shape: \n", output.shape)
        return output

    def backward(self, input, grad_input):
        if self.padding > 0:
            grad_input_padding = np.pad(array=grad_input, pad_width=(
                (0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant', constant_values=0)
            input_padding = np.pad(array=input, pad_width=(
                (0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant', constant_values=0)
        inverse_weights = np.rot90(self.weights, k=2, axes=(2, 3))
        '''
        compute dL/dx
        '''
        grad_output = np.zeros(input.shape)
        for channel in range(self.in_channels):
            for H in range(grad_output.shape[2]):
                for W in range(grad_output.shape[3]):
                    dot = np.multiply(grad_input_padding[:, :, H * self.strides: H*self.strides + self.kernel_size[0],
                                                         W*self.strides:W*self.strides+self.kernel_size[1]], inverse_weights[:, channel, :, :])
                    s = np.sum(dot, axis=(1, 2, 3))
                    grad_output[:, channel, H, W] = s

        '''
        compute dL/dw && dL/db
        '''
        grad_weights = np.zeros(self.weights.shape)

        for out_channel in range(grad_weights.shape[0]):
            for H in range(grad_weights.shape[2]):
                for W in range(grad_weights.shape[3]):
                    x = input_padding[:, :, H*self.strides: H * self.strides+self.kernel_size[0],
                                      W*self.strides: W*self.strides+self.kernel_size[1]]
                    y = grad_input[:, out_channel, H, W]
                    for i in range(x.shape[0]):
                        grad_weights[out_channel, :, :, :] += x[i]*y[i]
        grad_weights = grad_weights / input.shape[0]
        grad_biases = np.mean(grad_input, axis=0)
        grad_biases = np.sum(grad_biases, axis=(1, 2))
        self.weights = self.weights - self.lr*grad_weights
        self.biases = self.biases - self.lr*grad_biases
        return grad_output


class Flatten():
    def __init__(self):
        pass

    def forward(self, input):
        return input.reshape(input.shape[0], -1)

    def backward(self, input, grad_input):
        return grad_input.reshape(input.shape)


class Model():
    def __init__(self, lr=0.01):
        model = []
        model.append(Conv2d(in_channels=3, out_channels=6,
                            kernel_size=(3, 3), strides=1, padding=1, lr=lr))
        model.append(ReLU())
        model.append(Conv2d(in_channels=6, out_channels=2,
                            kernel_size=(4, 4), strides=1, padding=1, lr=lr))
        model.append(ReLU())
        model.append(Flatten())
        model.append(FC(1922, 128, lr=lr))
        model.append(ReLU())
        model.append(FC(128, 3, lr=lr))
        self.model = model

    def forward(self, input):
        activations = []
        activations.append(input.copy())
        for layer in self.model:
            activations.append(layer.forward(input))
            input = activations[-1]
        return activations

    def backward(self, loss_grad, layer_inputs):
        for layer_index in range(len(self.model))[:: -1]:
            loss_grad = self.model[layer_index].backward(
                layer_inputs[layer_index], loss_grad)

        return np.mean(loss_grad)
