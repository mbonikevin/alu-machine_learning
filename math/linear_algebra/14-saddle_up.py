#!/usr/bin/env python3
"""the module that defines np_matmul function"""

import numpy as np


def np_matmul(mat1, mat2):
    """matrix multiplication of two numpy arrays"""
    return np.matmul(np.array(mat1), np.array(mat2))
