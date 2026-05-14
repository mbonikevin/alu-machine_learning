#!/usr/bin/env python3
"""turn a vector of labels into a one hot matrix"""

import numpy as np


def one_hot_encode(Y, classes):
    """convert label vector Y into a one hot matrix of shape (classes, m)"""
    if type(Y) is not np.ndarray or type(classes) is not int:
        return None
    if len(Y.shape) != 1 or classes < 2 or classes <= np.max(Y):
        return None
    try:
        oh = np.zeros((classes, Y.shape[0]))
        oh[Y, np.arange(Y.shape[0])] = 1
        return oh
    except Exception:
        return None
