#!/usr/bin/env python3
"""neuron with forward propagation"""

import numpy as np


class Neuron:
    """one neuron, forward pass using sigmoid"""

    def __init__(self, nx):
        """set up the neuron"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """weights getter"""
        return self.__W

    @property
    def b(self):
        """bias getter"""
        return self.__b

    @property
    def A(self):
        """activation getter"""
        return self.__A

    def forward_prop(self, X):
        """run forward pass and store the activation"""
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A
