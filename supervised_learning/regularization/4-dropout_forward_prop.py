#!/usr/bin/env python3
"""this module conducts forward propagation using dropout"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """returns a dictionary with the outputs and dropout masks"""
    cache = {'A0': X}
    for i in range(1, L + 1):
        Z = np.matmul(weights['W' + str(i)], cache['A' + str(i - 1)]) \
            + weights['b' + str(i)]
        if i == L:
            t = np.exp(Z)
            cache['A' + str(i)] = t / np.sum(t, axis=0, keepdims=True)
        else:
            A = np.tanh(Z)
            D = np.random.binomial(1, keep_prob, size=A.shape)
            cache['D' + str(i)] = D
            cache['A' + str(i)] = A * D / keep_prob
    return cache
