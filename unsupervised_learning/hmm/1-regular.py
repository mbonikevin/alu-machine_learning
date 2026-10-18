#!/usr/bin/env python3
"""Determines the steady state probabilities of a regular markov chain."""
import numpy as np


def regular(P):
    """
    Determines the steady state probabilities of a regular markov chain.

    Args:
        P: square 2D numpy.ndarray of shape (n, n) representing the
            transition matrix. P[i, j] is the probability of transitioning
            from state i to state j and n is the number of states.

    Returns:
        A numpy.ndarray of shape (1, n) containing the steady state
        probabilities, or None on failure.
    """
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return None
    n = P.shape[0]
    if P.shape[1] != n:
        return None
    if np.any(P < 0):
        return None
    if not np.all(np.isclose(np.sum(P, axis=1), 1)):
        return None

    power = P.copy()
    for _ in range(n ** 2 + 1):
        if np.all(power > 0):
            break
        power = np.matmul(power, P)
    else:
        return None

    a = np.vstack((P.T - np.identity(n), np.ones((1, n))))
    b = np.append(np.zeros(n), 1)
    s = np.linalg.lstsq(a, b, rcond=None)[0]

    return s.reshape(1, n)
