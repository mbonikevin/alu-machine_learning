#!/usr/bin/env python3
"""deep neural network with private attrs"""

import numpy as np


class DeepNeuralNetwork:
    """deep net with private L, cache and weights"""

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
