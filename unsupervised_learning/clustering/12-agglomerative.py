#!/usr/bin/env python3
"""Performs agglomerative clustering on a dataset."""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    Performs agglomerative clustering with Ward linkage on a dataset and
    displays the dendrogram with each cluster in a different color.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset.
        dist: maximum cophenetic distance for all clusters.

    Returns:
        clss, a numpy.ndarray of shape (n,) containing the cluster indices
        for each data point.
    """
    links = scipy.cluster.hierarchy.linkage(X, method='ward')
    clss = scipy.cluster.hierarchy.fcluster(links, t=dist,
                                            criterion='distance')
    plt.figure()
    scipy.cluster.hierarchy.dendrogram(links, color_threshold=dist)
    plt.show()

    return clss
