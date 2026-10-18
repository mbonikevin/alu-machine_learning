#!/usr/bin/env python3
"""Determines if a markov chain is absorbing."""
import numpy as np


def absorbing(P):
    """
    Determines if a markov chain is absorbing.

    Args:
        P: square 2D numpy.ndarray of shape (n, n) representing the standard
            transition matrix. P[i, j] is the probability of transitioning
            from state i to state j and n is the number of states.

    Returns:
        True if it is absorbing, or False on failure.
    """
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return False
    n = P.shape[0]
    if P.shape[1] != n:
        return False
    if np.any(P < 0):
        return False
    if not np.all(np.isclose(np.sum(P, axis=1), 1)):
        return False

    absorbed = np.isclose(np.diagonal(P), 1)
    if not np.any(absorbed):
        return False
    if np.all(absorbed):
        return True

    reach = P > 0
    for _ in range(n):
        reach = reach | np.matmul(reach, P > 0)

    return bool(np.all(np.any(reach[:, absorbed], axis=1)))
