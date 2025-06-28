import numpy as np

class NeuralNetwork:

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

    def __init__(self, layer_sizes, learning_rate=0.01):
        if len(layer_sizes) < 2:
            raise ValueError(f"At least 2 layers required. {len(layer_sizes)} provided.")

        self.learning_rate = learning_rate
        self.num_incorrect = 0
        self.total = 0

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

    def _forward_pass(self, input):
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

        for i in range(1, len(self.layers)):
            layer = self.layers[i]
            layer.weights -= self.learning_rate * layer.grad_w
            layer.biases -= self.learning_rate * layer.grad_b

