#!/usr/bin/env python3
"""Calculates the most likely sequence of hidden states for a hidden markov
model."""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    Calculates the most likely sequence of hidden states for a hidden markov
    model.

    Args:
        Observation: numpy.ndarray of shape (T,) that contains the index of
            the observation. T is the number of observations.
        Emission: numpy.ndarray of shape (N, M) containing the emission
            probability of a specific observation given a hidden state.
            N is the number of hidden states and M the number of all possible
            observations.
        Transition: 2D numpy.ndarray of shape (N, N) containing the
            transition probabilities.
        Initial: numpy.ndarray of shape (N, 1) containing the probability of
            starting in a particular hidden state.

    Returns:
        path, P, or None, None on failure.
        path is a list of length T containing the most likely sequence of
        hidden states and P is the probability of obtaining the path
        sequence.
    """
    if not isinstance(Observation, np.ndarray) or len(Observation.shape) != 1:
        return None, None
    if not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2:
        return None, None
    N, M = Emission.shape
    if not isinstance(Transition, np.ndarray) or Transition.shape != (N, N):
        return None, None
    if not isinstance(Initial, np.ndarray) or Initial.shape != (N, 1):
        return None, None
    if not np.all(np.isclose(np.sum(Emission, axis=1), 1)):
        return None, None
    if not np.all(np.isclose(np.sum(Transition, axis=1), 1)):
        return None, None
    if not np.isclose(np.sum(Initial), 1):
        return None, None

    T = Observation.shape[0]
    V = np.zeros((N, T))
    back = np.zeros((N, T), dtype=int)
    V[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]

    for t in range(1, T):
        steps = V[:, t - 1, np.newaxis] * Transition
        back[:, t] = np.argmax(steps, axis=0)
        V[:, t] = np.max(steps, axis=0) * Emission[:, Observation[t]]

    last = np.argmax(V[:, T - 1])
    path = [int(last)]
    for t in range(T - 1, 0, -1):
        last = back[last, t]
        path.insert(0, int(last))

    return path, np.max(V[:, T - 1])
