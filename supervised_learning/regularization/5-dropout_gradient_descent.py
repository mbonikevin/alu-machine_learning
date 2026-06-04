#!/usr/bin/env python3
"""this module updates weights using gradient descent with dropout"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """updates the weights of the network in place"""
    m = Y.shape[1]
    dz = cache['A' + str(L)] - Y
    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]
        dw = (1 / m) * np.matmul(dz, A_prev.T)
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
        if i > 1:
            dz = np.matmul(W.T, dz) * (1 - A_prev ** 2)
            dz = dz * cache['D' + str(i - 1)] / keep_prob
        weights['W' + str(i)] = W - alpha * dw
        weights['b' + str(i)] = weights['b' + str(i)] - alpha * db
