#!/usr/bin/env python3
"""gated recurrent unit"""

import numpy as np


class GRUCell:
    """single step of a GRU"""

    def __init__(self, i, h, o):
        """set up the gate weights, the state weights and the output"""
        self.Wz = np.random.randn(i + h, h)
        self.Wr = np.random.randn(i + h, h)
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """run one time step and return the next state with the output"""
        concat = np.concatenate((h_prev, x_t), axis=1)
        z = sigmoid(np.matmul(concat, self.Wz) + self.bz)
        r = sigmoid(np.matmul(concat, self.Wr) + self.br)
        reset = np.concatenate((r * h_prev, x_t), axis=1)
        h_hat = np.tanh(np.matmul(reset, self.Wh) + self.bh)
        h_next = (1 - z) * h_prev + z * h_hat
        y = softmax(np.matmul(h_next, self.Wy) + self.by)
        return h_next, y


def sigmoid(x):
    """logistic activation"""
    return 1 / (1 + np.exp(-x))


def softmax(x):
    """softmax over the last axis"""
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)
