#!/usr/bin/env python3
"""single neuron for binary classification with private attrs"""

import numpy as np


class Neuron:
    """one neuron with private weights, bias and activation"""

    def __init__(self, nx):
        """set up the neuron with nx input features"""
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
