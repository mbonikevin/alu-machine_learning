#!/usr/bin/env python3
"""matrix inverse utilities"""


def determinant(mat):
    """compute determinant"""
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
    for c in range(n):
        sub = [r[:c] + r[c + 1:] for r in mat[1:]]
        total += ((-1) ** c) * mat[0][c] * determinant(sub)

    return total


def minor(mat):
    """return minor matrix"""
    if not all(isinstance(r, list) for r in mat):
        raise TypeError("matrix must be a list of lists")

    n = len(mat)
    if mat == [[]] or not all(len(r) == n for r in mat):
        raise ValueError("matrix must be a non-empty square matrix")

    if n == 1:
        return [[1]]

    result = []
    for i in range(n):
        row = []
        for j in range(n):
            sub = [r[:j] + r[j + 1:] for r in mat[:i] + mat[i + 1:]]
            row.append(determinant(sub))
        result.append(row)

    return result


def cofactor(mat):
    """return cofactor matrix"""
    n = len(mat)
    minors = minor(mat)

    cof = []
    for i in range(n):
        cof.append([
            minors[i][j] * ((-1) ** (i + j))
            for j in range(n)
        ])

    return cof


def adjugate(mat):
    """return adjugate matrix"""
    cof = cofactor(mat)
    n = len(cof)

    return [[cof[j][i] for j in range(n)] for i in range(n)]


def inverse(matrix):
    """return inverse or None"""
    if not all(isinstance(r, list) for r in matrix):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    if matrix == [[]] or not all(len(r) == n for r in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    det = determinant(matrix)
    if det == 0:
        return None

    adj = adjugate(matrix)
    return [[val / det for val in row] for row in adj]
