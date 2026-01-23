#!/usr/bin/env python3
"""the module that defines matrix_transpose function"""


def matrix_transpose(matrix):
    """returning the transpose of a 2D matrix"""
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]
