#!/usr/bin/env python3
"""deep neural network with a basic training loop"""

import numpy as np


class DeepNeuralNetwork:
    """deep net trainable with gradient descent"""

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
        """forward pass, fills the cache along the way"""
        self.__cache['A0'] = X
        for i in range(1, self.__L + 1):
            W = self.__weights['W' + str(i)]
            b = self.__weights['b' + str(i)]
            A_prev = self.__cache['A' + str(i - 1)]
            Z = np.matmul(W, A_prev) + b
            self.__cache['A' + str(i)] = 1 / (1 + np.exp(-Z))
        return self.__cache['A' + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """logistic regression cost"""
        m = Y.shape[1]
        loss = Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        return -np.sum(loss) / m

    def evaluate(self, X, Y):
        """forward then turn output into 0/1 labels"""
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
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

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """train for the given number of iterations"""
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        for _ in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(Y, self.__cache, alpha)
        return self.evaluate(X, Y)
