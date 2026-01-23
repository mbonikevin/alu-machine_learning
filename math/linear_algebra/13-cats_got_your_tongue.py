#!/usr/bin/env python3
"""the module that defines np_cat function"""

import numpy as np


def np_cat(mat1, mat2, axis=0):
    """concatenating two numpy arrays along tje axis"""
    return np.concatenate((np.array(mat1), np.array(mat2)), axis=axis)
