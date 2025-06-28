import csv
import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork

nn = NeuralNetwork([784, 16, 16, 10])

def display(row):
    plt.imshow(np.array(row[1:], dtype=np.uint8).reshape(28, 28), cmap='gray')
    plt.axis('off')
    plt.show()

EPOCHS = 3
for i in range(EPOCHS):
    print(f"Epoch: {i}/{EPOCHS}")
    with open('data/mnist_train.csv', 'r') as f:
        for j, row in enumerate(csv.reader(f)):
            label = int(row[0])
            pixels = np.array(row[1:], dtype=np.float32) / 255
            nn.train(pixels, label)

with open('data/mnist_test.csv', 'r') as f:
    for i, row in enumerate(csv.reader(f)):
        label = int(row[0])
        pixels = np.array(row[1:], dtype=np.float32) / 255
        nn.infer(pixels, label)

