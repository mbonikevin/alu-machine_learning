#!/usr/bin/env python3
"""Initializes variables for a Gaussian Mixture Model."""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Initializes variables for a Gaussian Mixture Model.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set.
        k: positive integer containing the number of clusters.

    Returns:
        pi, m, S, or None, None, None on failure.
        pi is a numpy.ndarray of shape (k,) containing the priors for each
        cluster, initialized evenly, m is a numpy.ndarray of shape (k, d)
        containing the centroid means initialized with K-means and S is a
        numpy.ndarray of shape (k, d, d) containing the covariance matrices,
        initialized as identity matrices.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None

    d = X.shape[1]
    pi = np.full((k,), 1 / k)
    m, _ = kmeans(X, k)
    S = np.tile(np.identity(d), (k, 1, 1))

    return pi, m, S
