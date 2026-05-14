#!/usr/bin/env python3
"""turn a one hot matrix back into a vector of labels"""

import numpy as np


def one_hot_decode(one_hot):
    """take one_hot of shape (classes, m) and return labels of shape (m,)"""
    if type(one_hot) is not np.ndarray or len(one_hot.shape) != 2:
        return None
    try:
        return np.argmax(one_hot, axis=0)
    except Exception:
        return None
