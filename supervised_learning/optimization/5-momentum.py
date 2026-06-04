#!/usr/bin/env python3
"""this module updates a variable using gradient descent with momentum"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """returns the updated variable and the new moment"""
    v = beta1 * v + (1 - beta1) * grad
    var = var - alpha * v
    return var, v
