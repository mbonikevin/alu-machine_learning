#!/usr/bin/env python3
"""this module shuffles the data points in two matrices the same way"""
import numpy as np


def shuffle_data(X, Y):
    """returns the shuffled X and Y matrices"""
    perm = np.random.permutation(X.shape[0])
    return X[perm], Y[perm]
