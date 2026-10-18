#!/usr/bin/env python3
"""Determines the probability of a markov chain being in a particular state
after a specified number of iterations."""
import numpy as np


def markov_chain(P, s, t=1):
    """
    Determines the probability of a markov chain being in a particular state
    after a specified number of iterations.

    Args:
        P: square 2D numpy.ndarray of shape (n, n) representing the
            transition matrix. P[i, j] is the probability of transitioning
            from state i to state j and n is the number of states.
        s: numpy.ndarray of shape (1, n) representing the probability of
            starting in each state.
        t: number of iterations that the markov chain has been through.

    Returns:
        A numpy.ndarray of shape (1, n) representing the probability of being
        in a specific state after t iterations, or None on failure.
    """
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return None
    n = P.shape[0]
    if P.shape[1] != n:
        return None
    if not isinstance(s, np.ndarray) or s.shape != (1, n):
        return None
    if not isinstance(t, int) or t < 0:
        return None
    if not np.all(np.isclose(np.sum(P, axis=1), 1)):
        return None

    state = s
    for _ in range(t):
        state = np.matmul(state, P)

    return state
