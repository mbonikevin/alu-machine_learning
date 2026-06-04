#!/usr/bin/env python3
"""this module updates a variable using the rmsprop algorithm"""


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """returns the updated variable and the new moment"""
    s = beta2 * s + (1 - beta2) * grad ** 2
    var = var - alpha * grad / (s ** 0.5 + epsilon)
    return var, s
