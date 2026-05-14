#!/usr/bin/env python3
"""deep neural network for multiclass classification with softmax"""

import pickle
import numpy as np
import matplotlib.pyplot as plt


class DeepNeuralNetwork:
    """deep net with softmax output and categorical cross entropy cost"""

    def __init__(self, nx, layers):
        """build the network from a list of layer sizes"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for i in range(self.__L):
            if type(layers[i]) is not int or layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")
            prev = nx if i == 0 else layers[i - 1]
            self.__weights['W' + str(i + 1)] = (
                np.random.randn(layers[i], prev) * np.sqrt(2 / prev)
            )
            self.__weights['b' + str(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """layer count"""
        return self.__L

    @property
    def cache(self):
        """forward pass cache"""
        return self.__cache

    @property
    def weights(self):
        """weights and biases"""
        return self.__weights

    def forward_prop(self, X):
        """forward pass, sigmoid for hidden layers, softmax for output"""
        self.__cache['A0'] = X
        for i in range(1, self.__L + 1):
            W = self.__weights['W' + str(i)]
            b = self.__weights['b' + str(i)]
            A_prev = self.__cache['A' + str(i - 1)]
            Z = np.matmul(W, A_prev) + b
            if i == self.__L:
                t = np.exp(Z)
                self.__cache['A' + str(i)] = (
                    t / np.sum(t, axis=0, keepdims=True)
                )
            else:
                self.__cache['A' + str(i)] = 1 / (1 + np.exp(-Z))
        return self.__cache['A' + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """categorical cross entropy cost"""
        m = Y.shape[1]
        return -np.sum(Y * np.log(A)) / m

    def evaluate(self, X, Y):
        """forward then return one hot predictions and cost"""
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.zeros_like(A, dtype=int)
        prediction[np.argmax(A, axis=0), np.arange(A.shape[1])] = 1
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """one backprop pass through every layer"""
        m = Y.shape[1]
        L = self.__L
        dZ = cache['A' + str(L)] - Y
        for i in range(L, 0, -1):
            A_prev = cache['A' + str(i - 1)]
            dW = np.matmul(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            W = self.__weights['W' + str(i)]
            if i > 1:
                dZ = np.matmul(W.T, dZ) * A_prev * (1 - A_prev)
            self.__weights['W' + str(i)] = W - alpha * dW
            self.__weights['b' + str(i)] = (
                self.__weights['b' + str(i)] - alpha * db
            )

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """train the net and optionally report or plot the cost"""
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        costs = []
        steps = []
        for i in range(iterations + 1):
            A, _ = self.forward_prop(X)
            if i % step == 0 or i == iterations:
                c = self.cost(Y, A)
                if verbose:
                    print("Cost after {} iterations: {}".format(i, c))
                if graph:
                    costs.append(c)
                    steps.append(i)
            if i < iterations:
                self.gradient_descent(Y, self.__cache, alpha)
        if graph:
            plt.plot(steps, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()
        return self.evaluate(X, Y)

    def save(self, filename):
        """pickle this instance to disk, add .pkl if missing"""
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """load a pickled DeepNeuralNetwork from disk"""
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None
