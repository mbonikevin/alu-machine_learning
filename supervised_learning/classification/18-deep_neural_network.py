#!/usr/bin/env python3
"""deep neural network with forward propagation"""

import numpy as np


class DeepNeuralNetwork:
    """deep net, forward pass with sigmoid"""

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
