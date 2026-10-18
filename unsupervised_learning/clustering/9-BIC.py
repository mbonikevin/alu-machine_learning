#!/usr/bin/env python3
"""Finds the best number of clusters for a GMM using the Bayesian
Information Criterion."""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Finds the best number of clusters for a GMM using the Bayesian
    Information Criterion.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set.
        kmin: positive integer containing the minimum number of clusters to
            check for (inclusive).
        kmax: positive integer containing the maximum number of clusters to
            check for (inclusive).
        iterations: positive integer containing the maximum number of
            iterations for the EM algorithm.
        tol: non-negative float containing the tolerance for the EM
            algorithm.
        verbose: boolean that determines if the EM algorithm should print
            information to the standard output.

    Returns:
        best_k, best_result, l, b, or None, None, None, None on failure.
        best_k is the best value for k based on its BIC, best_result is a
        tuple containing pi, m, S for that k, l contains the log likelihood
        for each cluster size tested and b contains the BIC value for each
        cluster size tested.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None, None, None
    if kmax is not None and (not isinstance(kmax, int) or kmax <= 0):
        return None, None, None, None
    if kmax is None:
        kmax = X.shape[0]
    if kmax - kmin < 1:
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    n, d = X.shape
    results = []
    likelihoods = []
    bics = []

    for k in range(kmin, kmax + 1):
        pi, m, S, _, likelihood = expectation_maximization(
            X, k, iterations, tol, verbose)
        results.append((pi, m, S))
        likelihoods.append(likelihood)
        p = (k * d * (d + 1) / 2) + (d * k) + k - 1
        bics.append(p * np.log(n) - 2 * likelihood)

    likelihoods = np.array(likelihoods)
    bics = np.array(bics)
    best = np.argmin(bics)

    return best + kmin, results[best], likelihoods, bics
