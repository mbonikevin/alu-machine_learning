#!/usr/bin/env python3
"""Performs K-means on a dataset."""
import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset.
        k: positive integer containing the number of clusters.
        iterations: positive integer containing the maximum number of
            iterations that should be performed.

    Returns:
        C, clss, or None, None on failure.
        C is a numpy.ndarray of shape (k, d) containing the centroid means
        for each cluster and clss is a numpy.ndarray of shape (n,) containing
        the index of the cluster in C that each data point belongs to.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape
    low = np.min(X, axis=0)
    high = np.max(X, axis=0)
    C = np.random.uniform(low, high, size=(k, d))

    for _ in range(iterations):
        C_prev = np.copy(C)
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=-1)
        clss = np.argmin(distances, axis=-1)

        for i in range(k):
            if not np.any(clss == i):
                C[i] = np.random.uniform(low, high, size=(1, d))
            else:
                C[i] = np.mean(X[clss == i], axis=0)

        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=-1)
        clss = np.argmin(distances, axis=-1)

        if np.all(C_prev == C):
            return C, clss

    return C, clss
