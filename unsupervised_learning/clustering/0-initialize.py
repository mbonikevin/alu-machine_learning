#!/usr/bin/env python3
"""Initializes cluster centroids for K-means."""
import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset that will be
            used for K-means clustering. n is the number of data points and d
            is the number of dimensions for each data point.
        k: positive integer containing the number of clusters.

    Returns:
        A numpy.ndarray of shape (k, d) containing the initialized centroids
        for each cluster, or None on failure.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    d = X.shape[1]
    low = np.min(X, axis=0)
    high = np.max(X, axis=0)

    return np.random.uniform(low, high, size=(k, d))
