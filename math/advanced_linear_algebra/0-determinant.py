#!/usr/bin/env python3
"""matrix determinant helper"""


def determinant(mat):
    """return determinant"""
    if mat == [[]]:
        return 1

    if not all(isinstance(r, list) for r in mat):
        raise TypeError("matrix must be a list of lists")

    n = len(mat)
    if not all(len(r) == n for r in mat):
        raise ValueError("matrix must be a square matrix")

    if n == 1:
        return mat[0][0]

    if n == 2:
        return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]

    total = 0
    for col in range(n):
        sub = [r[:col] + r[col + 1:] for r in mat[1:]]
        total += ((-1) ** col) * mat[0][col] * determinant(sub)

    return total
