#!/usr/bin/env python3
"""cell of a simple recurrent neural network"""

import numpy as np


class RNNCell:
    """single step of a vanilla RNN"""

    def __init__(self, i, h, o):
        """set up the weights for the hidden state and the output"""
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """run one time step and return the next state with the output"""
        concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(concat, self.Wh) + self.bh)
        y = np.matmul(h_next, self.Wy) + self.by
        y = np.exp(y - np.max(y, axis=1, keepdims=True))
        y = y / np.sum(y, axis=1, keepdims=True)
        return h_next, y
