#!/usr/bin/env python3
"""Performs the expectation maximization for a GMM."""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Performs the expectation maximization for a GMM.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set.
        k: positive integer containing the number of clusters.
        iterations: positive integer containing the maximum number of
            iterations for the algorithm.
        tol: non-negative float containing the tolerance of the log
            likelihood, used to determine early stopping.
        verbose: boolean that determines if the function should print
            information about the algorithm every 10 iterations and after the
            last iteration.

    Returns:
        pi, m, S, g, l, or None, None, None, None, None on failure.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    g, likelihood = expectation(X, pi, m, S)
    i = 0

    for i in range(iterations):
        if verbose and i % 10 == 0:
            print('Log Likelihood after {} iterations: {}'.format(
                i, round(likelihood, 5)))
        pi, m, S = maximization(X, g)
        g, new_likelihood = expectation(X, pi, m, S)
        converged = abs(new_likelihood - likelihood) <= tol
        likelihood = new_likelihood
        if converged:
            break

    if verbose:
        print('Log Likelihood after {} iterations: {}'.format(
            i + 1, round(likelihood, 5)))

    return pi, m, S, g, likelihood
