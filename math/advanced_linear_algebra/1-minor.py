#!/usr/bin/env python3
"""matrix minor utilities"""


def determinant(mat):
    """return determinant of a square matrix"""
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

    det = 0
    for col in range(n):
        sign = -1 if col % 2 else 1
        sub = [r[:col] + r[col + 1:] for r in mat[1:]]
        det += sign * mat[0][col] * determinant(sub)

    return det


def minor(matrix):
    """compute matrix minor"""
    if not all(isinstance(r, list) for r in matrix):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    if matrix == [[]] or not all(len(r) == n for r in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    if n == 1:
        return [[1]]

    result = []
    for i in range(n):
        row = []
        for j in range(n):
            sub = [
                r[:j] + r[j + 1:]
                for r in (matrix[:i] + matrix[i + 1:])
            ]
            row.append(determinant(sub))
        result.append(row)

    return result
