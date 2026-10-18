#!/usr/bin/env python3
"""Calculates the maximization step in the EM algorithm for a GMM."""
import numpy as np


def maximization(X, g):
    """
    Calculates the maximization step in the EM algorithm for a GMM.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set.
        g: numpy.ndarray of shape (k, n) containing the posterior
            probabilities for each data point in each cluster.

    Returns:
        pi, m, S, or None, None, None on failure.
        pi is a numpy.ndarray of shape (k,) containing the updated priors,
        m is a numpy.ndarray of shape (k, d) containing the updated centroid
        means and S is a numpy.ndarray of shape (k, d, d) containing the
        updated covariance matrices.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None

    n, d = X.shape
    k = g.shape[0]
    if g.shape[1] != n:
        return None, None, None
    if not np.all(np.isclose(np.sum(g, axis=0), 1)):
        return None, None, None

    totals = np.sum(g, axis=1)
    pi = totals / n
    m = np.matmul(g, X) / totals[:, np.newaxis]

    S = np.zeros((k, d, d))
    for i in range(k):
        diff = X - m[i]
        S[i] = np.matmul(g[i] * diff.T, diff) / totals[i]

    return pi, m, S
