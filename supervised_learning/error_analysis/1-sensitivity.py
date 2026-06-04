#!/usr/bin/env python3
"""this module calculates the sensitivity for each class"""
import numpy as np


def sensitivity(confusion):
    """returns the sensitivity of each class"""
    return np.diag(confusion) / np.sum(confusion, axis=1)
