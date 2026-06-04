#!/usr/bin/env python3
"""this module calculates the precision for each class"""
import numpy as np


def precision(confusion):
    """returns the precision of each class"""
    return np.diag(confusion) / np.sum(confusion, axis=0)
