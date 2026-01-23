#!/usr/bin/env python3
"""the module that defines matrix_shape function"""


def matrix_shape(matrix):
    """Returning the shape of a matrix as a list of integers"""
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
