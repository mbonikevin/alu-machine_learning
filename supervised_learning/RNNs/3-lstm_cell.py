#!/usr/bin/env python3
"""long short term memory unit"""

import numpy as np


class LSTMCell:
    """single step of an LSTM"""

    def __init__(self, i, h, o):
        """set up the gate weights, the cell state and the output"""
        self.Wf = np.random.randn(i + h, h)
        self.Wu = np.random.randn(i + h, h)
        self.Wc = np.random.randn(i + h, h)
        self.Wo = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """run one time step and return both states with the output"""
        concat = np.concatenate((h_prev, x_t), axis=1)
        f = sigmoid(np.matmul(concat, self.Wf) + self.bf)
        u = sigmoid(np.matmul(concat, self.Wu) + self.bu)
        c_hat = np.tanh(np.matmul(concat, self.Wc) + self.bc)
        c_next = f * c_prev + u * c_hat
        o = sigmoid(np.matmul(concat, self.Wo) + self.bo)
        h_next = o * np.tanh(c_next)
        y = softmax(np.matmul(h_next, self.Wy) + self.by)
        return h_next, c_next, y


def sigmoid(x):
    """logistic activation"""
    return 1 / (1 + np.exp(-x))


def softmax(x):
    """softmax over the last axis"""
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)
