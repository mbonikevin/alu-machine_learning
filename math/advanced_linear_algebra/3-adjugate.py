#!/usr/bin/env python3
"""adjugate matrix operations"""


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
    for c in range(n):
        sub = [r[:c] + r[c + 1:] for r in mat[1:]]
        total += ((-1) ** c) * mat[0][c] * determinant(sub)

    return total


def minor(matrix):
    """return minor matrix"""
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
            sub = [r[:j] + r[j + 1:] for r in matrix[:i] + matrix[i + 1:]]
            row.append(determinant(sub))
        result.append(row)

    return result


def cofactor(matrix):
    """return cofactor matrix"""
    if not all(isinstance(r, list) for r in matrix):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    if matrix == [[]] or not all(len(r) == n for r in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    if n == 1:
        return [[1]]

    minors = minor(matrix)
    cof = []

    for i in range(n):
        row = []
        for j in range(n):
            row.append(minors[i][j] * ((-1) ** (i + j)))
        cof.append(row)

    return cof


def adjugate(matrix):
    """return adjugate matrix"""
    if not all(isinstance(r, list) for r in matrix):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    if matrix == [[]] or not all(len(r) == n for r in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    if n == 1:
        return [[1]]

    cof = cofactor(matrix)
    adj = []

    for j in range(n):
        adj.append([cof[i][j] for i in range(n)])

    return adj
