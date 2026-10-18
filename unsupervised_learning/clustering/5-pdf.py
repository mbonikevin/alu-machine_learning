#!/usr/bin/env python3
"""Calculates the probability density function of a Gaussian distribution."""
import numpy as np


def pdf(X, m, S):
    """
    Calculates the probability density function of a Gaussian distribution.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data points whose PDF
            should be evaluated.
        m: numpy.ndarray of shape (d,) containing the mean of the
            distribution.
        S: numpy.ndarray of shape (d, d) containing the covariance of the
            distribution.

    Returns:
        P, or None on failure. P is a numpy.ndarray of shape (n,) containing
        the PDF values for each data point, with a minimum value of 1e-300.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None
    d = X.shape[1]
    if m.shape[0] != d or S.shape[0] != d or S.shape[1] != d:
        return None

    det = np.linalg.det(S)
    inv = np.linalg.inv(S)
    norm = 1 / np.sqrt(((2 * np.pi) ** d) * det)
    diff = X - m
    expo = -0.5 * np.sum(np.matmul(diff, inv) * diff, axis=1)
    P = norm * np.exp(expo)

    return np.maximum(P, 1e-300)
