import math
import random
import numpy as np

class LayerLink:
    def __init__(self, prev_node_count, node_count):
        self.weights = (np.random.random((prev_node_count, node_count)) * 2) - 1
        self.bias = (np.random.random((1, node_count)) * 2) - 1


class NeuralNetwork:
    def __init__(self, layers):
        self.id = random.random()
        self.fitness = 0
        self.weights_flat = np.array([])
        self.bias_flat = np.array([])
        self.learning_rate = 0.1
        self.layer_count = len(layers) - 1
        self.inputs = layers[0]
        self.output_nodes = layers[-1]
        self.layer_link = []
        for i in range(self.layer_count):
            self.layer_link.append(LayerLink(layers[i], layers[i+1]))
            self.weights_flat = np.concatenate((self.weights_flat, self.layer_link[i].weights.flatten()))
            self.bias_flat = np.concatenate((self.bias_flat,self.layer_link[i].bias.flatten()))

    @staticmethod
    def activation(x):
        return 1/(1+math.exp(-x))

    @staticmethod
    def derivative(y):
        return y * (1 - y)

    @staticmethod
    def relu(x):
        return max(0, x)

    def predict(self, input):
        result = np.array(input)
        for layers in self.layer_link:
            result = np.dot(result, layers.weights)
            result += layers.bias
            result = np.vectorize(self.relu)(result)

        return result

    def reconstruct_weights(self):
        start = 0
        for layer in self.layer_link:
            shape = layer.weights.shape
            total = shape[0] * shape[1]
            layer.weights = self.weights_flat[start:start+total].reshape(shape)
            start += total

    def reconstruct_bias(self):
        start = 0
        for layer in self.layer_link:
            shape = layer.bias.shape
            total = shape[0] * shape[1]
            layer.bias = self.bias_flat[start:start+total].reshape(shape)
            start += total


