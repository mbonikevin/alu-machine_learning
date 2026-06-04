#!/usr/bin/env python3
"""this module calculates the cost of a network with l2 regularization"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """returns the cost accounting for l2 regularization"""
    norm = 0
    for i in range(1, L + 1):
        norm += np.linalg.norm(weights['W' + str(i)]) ** 2
    return cost + (lambtha / (2 * m)) * norm
