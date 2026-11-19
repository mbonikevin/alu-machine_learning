#!/usr/bin/env python3
"""Simple policy function and Monte-Carlo policy gradient"""
import numpy as np


def policy(matrix, weight):
    """Computes the softmax policy of matrix with a weight of weight"""
    z = np.exp(matrix.dot(weight))
    return z / np.sum(z, axis=1, keepdims=True)


def policy_gradient(state, weight):
    """
    Computes the Monte-Carlo policy gradient

    state: matrix representing the current observation of the environment
    weight: matrix of random weight

    Returns: the action and the gradient (in this order)
    """
    probs = policy(state, weight)
    action = np.random.choice(probs.shape[1], p=probs[0])
    # d log(softmax[action]) / dz = onehot(action) - softmax
    dlog = -probs
    dlog[0, action] += 1
    return action, state.T.dot(dlog)
