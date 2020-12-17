import numpy as np


class FC():
    def __init__(self, input, output, lr=0.1):
        self.input = input
        self.output = output
        self.lr = lr
        self.weights = np.random.normal(loc=0.0, scale=np.sqrt(2/self.input + self.output),
                                        size=(self.input, self.output))
        self.biases = np.zeros(output)

    def forward(self, input):
        return np.dot(input, self.weights) + self.biases

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

    def backward(self, input, grad_output):
        relu_grad = input > 0
        return grad_output*relu_grad


class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, strides=1, padding=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.weights = self.weights = np.random.normal(loc=0.0, scale=np.sqrt(2/self.in_channels + self.out_channels),
                                                       size=(self.out_channels, self.in_channels, kernel_size[0], kernel_size[1]))
        #self.biases = np.zeros(out_channels, in_channels)
        pass

    def forward(self, input):
        output_shape_x = int(
            ((input.shape[1] - self.kernel_size[0] + 2*self.padding) / self.strides) + 1)
        output_shape_y = int(
            ((input.shape[2] - self.kernel_size[1] + 2*self.padding) / self.strides) + 1)

        output = np.zeros((input.shape[0], output_shape_x, output_shape_y))
        pass
        # for y in range(output_shape_y):
        #     for x in range(output_shape_x):

        #         output[input.shape[0], x, y] = np.sum(np.dot())

    def backward(self):
        pass


class Model():
    def __init__(self):
        model = []
        model.append(Conv2d(in_channels=3, out_channels=16,
                            kernel_size=(1, 1), strides=1, padding=0))
        self.model = model

    def forward(self, input):
        activations = []
        activations.append(input.copy())
        for layer in self.model:
            activations.append(layer.forward(input))
            input = activations[-1]
        return activations

    def backward(self, layer_inputs, grad_input):
        for layer_index in range(len(self.model))[::-1]:
            loss_grad = self.model[layer_index].backward(
                layer_inputs[layer_index], loss_grad)

        return np.mean(loss_grad)
