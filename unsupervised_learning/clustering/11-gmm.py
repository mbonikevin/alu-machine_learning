#!/usr/bin/env python3
"""Calculates a GMM from a dataset using sklearn."""
import sklearn.mixture


def gmm(X, k):
    """
    Calculates a GMM from a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset.
        k: number of clusters.

    Returns:
        pi, m, S, clss, bic.
        pi is a numpy.ndarray of shape (k,) containing the cluster priors,
        m is a numpy.ndarray of shape (k, d) containing the centroid means,
        S is a numpy.ndarray of shape (k, d, d) containing the covariance
        matrices, clss is a numpy.ndarray of shape (n,) containing the
        cluster indices for each data point and bic is a numpy.ndarray of
        shape (kmax - kmin + 1) containing the BIC value for each cluster
        size tested.
    """
    model = sklearn.mixture.GaussianMixture(n_components=k).fit(X)

    return (model.weights_, model.means_, model.covariances_,
            model.predict(X), model.bic(X))
