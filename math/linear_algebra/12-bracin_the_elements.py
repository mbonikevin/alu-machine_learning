#!/usr/bin/env python3
"""the module that defines np_elementwise function"""

import numpy as np


def np_elementwise(mat1, mat2):
    """performing element-wise addition, subtraction, multiplication, and division"""
    mat1 = np.array(mat1)
    mat2 = np.array(mat2)
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
