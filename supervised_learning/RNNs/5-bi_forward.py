#!/usr/bin/env python3
"""cell of a bidirectional RNN"""

import numpy as np


class BidirectionalCell:
    """single step of a bidirectional RNN"""

    def __init__(self, i, h, o):
        """set up the weights for both directions and the output"""
        self.Whf = np.random.randn(i + h, h)
        self.Whb = np.random.randn(i + h, h)
        self.Wy = np.random.randn(2 * h, o)
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """run one time step in the forward direction"""
        concat = np.concatenate((h_prev, x_t), axis=1)
        return np.tanh(np.matmul(concat, self.Whf) + self.bhf)
