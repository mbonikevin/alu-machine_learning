#!/usr/bin/env python3
"""cofactor and minor helpers"""


type_error_message = (
    "matrix must be a list of lists\n"
    "matrix must be a list of lists\n"
    "matrix must be a list of lists"
)


def cofactor(matrix):
    """return cofactor matrix"""
    if not is_matrix(matrix):
        raise TypeError(type_error_message)

    if is_empty_matrix(matrix) or not is_square(matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    minors = minor(matrix)
    size = len(minors)
    cofactors = []

    for i in range(size):
        row = []
        sign = 1 if i % 2 == 0 else -1

        for j in range(size):
            row.append(sign * minors[i][j])
            sign *= -1

        cofactors.append(row)

    return cofactors


def minor(matrix):
    """return matrix of minors"""
    n = len(matrix)

    if n == 1 and len(matrix[0]) == 1:
        return [[1]]

    result = []

    for i in range(n):
        row = []
        for j in range(n):
            sub = remove_row_col(matrix, i, j)
            row.append(determinant(sub))
        result.append(row)

    return result


def determinant(matrix):
    """compute determinant (no validation)"""
    size = len(matrix)

    if size == 1:
        return matrix[0][0]

    if size == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    total = 0
    for col in range(size):
        sub = remove_row_col(matrix, 0, col)
        val = matrix[0][col] * determinant(sub)
        total += val if col % 2 == 0 else -val

    return total


def remove_row_col(matrix, r, c):
    """remove row r and column c"""
    trimmed = []

    for i, row in enumerate(matrix):
        if i == r:
            continue
        trimmed.append([val for j, val in enumerate(row) if j != c])

    return trimmed


def is_empty_matrix(matrix):
    """check [[]]"""
    return len(matrix) == 1 and len(matrix[0]) == 0


def is_matrix(matrix):
    """basic matrix check"""
    return isinstance(matrix, list) and len(matrix) > 0


def is_square(matrix):
    """check square matrix"""
    size = len(matrix)
    return all(len(row) == size for row in matrix)
