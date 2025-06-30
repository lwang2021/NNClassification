import math
import numpy as np

class NeuralNetwork:

    class Conv:
        # Size is width/height, 
        def __init__(self):
            self.size = None

            self.kernel = np.random.randn(3, 3) * 0.1
            self.bias = 0
            self.input = None
            self.preactivation = None
            self.activation = None

    class Layer:
        def __init__(self, size, last_size):
            self.size = size

            # This is the input layer
            if last_size is None:
                return

            # He initialization
            self.weights = np.random.randn(size, last_size) * np.sqrt(2 / last_size)
            self.biases = np.zeros(size)

            self.preactivation = None
            self.activation = None
            self.grad_w = None
            self.grad_b = None

    def __init__(self, layer_sizes, learning_rate=0.01, doConvolution = False, convLayers = 1):
        if len(layer_sizes) < 2:
            raise ValueError(f"At least 2 NN layers required. {len(layer_sizes)} provided.")
        
        self.convLayers = []
        if doConvolution:
            if convLayers < 1:
                raise ValueError(f"At least 1 Convolution layers required. {convLayers} provided.")
            else:
                for _ in range(convLayers):
                    self.convLayers.append(self.Conv())

        self.learning_rate = learning_rate
        self.num_incorrect = 0
        self.total = 0
        self.doConvolution = doConvolution

        self.layers = []
        last = None
        for size in layer_sizes:
            self.layers.append(self.Layer(size, last))
            last = size

    @staticmethod
    def softmax(z):
        exp_z = np.exp(z - np.max(z))
        return exp_z / np.sum(exp_z)

    @staticmethod
    def relu(z):
        return np.maximum(0, z)
    
    def conv2d(self, convLayer, padding=1, stride=1):
        image = convLayer.input
        H, W = image.shape
        kH, kW = convLayer.kernel.shape

        padded = np.pad(image, pad_width=padding, mode='constant', constant_values=0)

        out_H = (H + 2 * padding - kH) // stride + 1
        out_W = (W + 2 * padding - kW) // stride + 1
        output = np.zeros((out_H, out_W))

        for i in range(out_H):
            for j in range(out_W):
                region = padded[i*stride:i*stride + kH, j*stride:j*stride + kW]
                output[i, j] = np.sum(region * convLayer.kernel) + convLayer.bias
        return output
    
    def conv2d_backward(self, d_out, input, kernel, padding=1, stride=1):

        kH, kW = kernel.shape
        H_in, W_in = input.shape
        H_out, W_out = d_out.shape

        padded_input = np.pad(input, padding, mode='constant', constant_values=0)
        d_input_padded = np.zeros_like(padded_input, dtype=np.float32)
        d_kernel = np.zeros_like(kernel)
        d_bias = 0

        for i in range(H_out):
            for j in range(W_out):
                region = padded_input[i*stride : i*stride+kH, j*stride : j*stride+kW]

                d_kernel += d_out[i, j] * region

                d_input_padded[i*stride : i*stride+kH, j*stride : j*stride+kW] += d_out[i, j] * kernel

        d_bias = np.sum(d_out)

        if padding > 0:
            d_input = d_input_padded[padding:-padding, padding:-padding]
        else:
            d_input = d_input_padded

        return d_input, d_kernel, d_bias
    
    def max_pool2d(self, x, pool_size=2, stride=2):
        H, W = x.shape
        out_H = (H - pool_size) // stride + 1
        out_W = (W - pool_size) // stride + 1
        pooled = np.zeros((out_H, out_W))

        for i in range(out_H):
            for j in range(out_W):
                region = x[i*stride:i*stride+pool_size, j*stride:j*stride+pool_size]
                pooled[i, j] = np.max(region)
        return pooled
    
    def max_pool2d_backward(self, d_out, input, pool_size=2, stride=2):
        H_in, W_in = input.shape
        H_out, W_out = d_out.shape
        d_input = np.zeros_like(input)

        for i in range(H_out):
            for j in range(W_out):
                h_start = i * stride
                w_start = j * stride
                region = input[h_start:h_start+pool_size, w_start:w_start+pool_size]

                # Mask: 1 at max value, 0 elsewhere
                mask = (region == np.max(region))
                d_input[h_start:h_start+pool_size, w_start:w_start+pool_size] += d_out[i, j] * mask

        return d_input


    def _forward_pass(self, input):
        if self.doConvolution:
            for i in range(len(self.convLayers)):
                self.convLayers[i].input = input
                conv_output = self.conv2d(self.convLayers[i], padding=1, stride=1)
                self.convLayers[i].preactivation = conv_output
                relu_output = self.relu(conv_output)
                self.convLayers[i].activation = relu_output
                input = self.max_pool2d(relu_output, pool_size=2, stride=2)

        input = np.array(input).flatten()
        self.layers[0].activation = input

        for i in range(1, len(self.layers)):
            layer = self.layers[i]
            prev_layer = self.layers[i - 1]
            layer.preactivation = layer.weights @ prev_layer.activation + layer.biases
            if i == len(self.layers) - 1:
                layer.activation = self.softmax(layer.preactivation)
            else:
                layer.activation = self.relu(layer.preactivation)

    def infer(self, pixels, label):
        self._forward_pass(pixels)
        guess = np.argmax(self.layers[-1].preactivation)

        self.total += 1
        emoji = '✅'
        if not label == guess:
            self.num_incorrect += 1
            emoji = '❌'
        accuracy = (1 - self.num_incorrect / self.total) * 100
        print(f"Expected: {label} | Actual: {guess} | Accuracy: {accuracy}% {emoji}")

        return accuracy

    def train(self, pixels, label):
        self._forward_pass(pixels)

        output_layer = self.layers[-1]
        target = np.zeros(output_layer.size)
        target[label] = 1
        print('Loss:', -np.log(output_layer.activation[label] + 1e-9))  # cross-entropy

        grad_z = output_layer.activation - target
        for i in range(len(self.layers) - 1, 0, -1):
            layer = self.layers[i]
            prev_layer = self.layers[i - 1]

            layer.grad_w = np.outer(grad_z, prev_layer.activation)
            layer.grad_b = grad_z

            if i > 1:
                grad_z = (layer.weights.T @ grad_z) * (prev_layer.preactivation > 0)
            if i == 1 and self.doConvolution:
                grad_z = (layer.weights.T @ grad_z).reshape(prev_layer.activation.shape)
                grad_z = grad_z * (prev_layer.activation > 0)


        for i in range(1, len(self.layers)):
            layer = self.layers[i]
            layer.weights -= self.learning_rate * layer.grad_w
            layer.biases -= self.learning_rate * layer.grad_b

        if self.doConvolution:

            grad_z = grad_z.reshape(int(math.sqrt(grad_z.size)), 
                                    int(math.sqrt(grad_z.size)))
            
            for i in range(len(self.convLayers) - 1, -1, -1):
                conv_layer = self.convLayers[i]

                grad_z = self.max_pool2d_backward(
                    d_out=grad_z, 
                    input=conv_layer.activation, 
                    pool_size=2, 
                    stride=2)
                
                grad_z = grad_z * (conv_layer.preactivation > 0)
                
                grad_z, d_kernel, d_bias = self.conv2d_backward(
                    d_out=grad_z,
                    input=conv_layer.input,    
                    kernel=conv_layer.kernel,
                    padding=1,
                    stride=1
                )

                conv_layer.kernel -= self.learning_rate * d_kernel
                conv_layer.bias -= self.learning_rate * d_bias

