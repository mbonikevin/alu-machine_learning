#!/usr/bin/env python3
"""matrix definiteness check"""
import numpy as np


def definiteness(mat):
    """return matrix definiteness"""
    if not isinstance(mat, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    # must be symmetric
    if not np.array_equal(mat, mat.T):
        return None

    eigs = np.linalg.eigvals(mat)

    if np.all(eigs > 0):
        return "Positive definite"

    if np.all(eigs >= 0):
        return "Positive semi-definite"

    if np.all(eigs < 0):
        return "Negative definite"

    if np.all(eigs <= 0):
        return "Negative semi-definite"

    return "Indefinite"
