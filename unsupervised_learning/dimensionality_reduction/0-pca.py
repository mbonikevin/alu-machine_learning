#!/usr/bin/env python3
"""Performs principal components analysis on a dataset."""
import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) where n is the number of data points
            and d is the number of dimensions in each point. All dimensions
            have a mean of 0 across all data points.
        var: fraction of the variance that the PCA transformation should
            maintain.

    Returns:
        The weights matrix W, a numpy.ndarray of shape (d, nd) that maintains
        var fraction of X's original variance, where nd is the new
        dimensionality of the transformed X.
    """
    _, s, vh = np.linalg.svd(X)
    ratios = np.cumsum(s) / np.sum(s)
    nd = np.argwhere(ratios >= var)[0, 0]

    return vh[:nd + 1].T
