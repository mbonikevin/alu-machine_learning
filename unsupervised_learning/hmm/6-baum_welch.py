#!/usr/bin/env python3
"""Performs the Baum-Welch algorithm for a hidden markov model."""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    Performs the forward algorithm for a hidden markov model.

    Args:
        Observation: numpy.ndarray of shape (T,) containing the index of the
            observation.
        Emission: numpy.ndarray of shape (M, N) containing the emission
            probabilities.
        Transition: numpy.ndarray of shape (M, M) containing the transition
            probabilities.
        Initial: numpy.ndarray of shape (M, 1) containing the starting
            probabilities.

    Returns:
        F, a numpy.ndarray of shape (M, T) containing the forward path
        probabilities.
    """
    T = Observation.shape[0]
    M = Emission.shape[0]
    F = np.zeros((M, T))
    F[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]

    for t in range(1, T):
        F[:, t] = np.matmul(F[:, t - 1], Transition) * \
            Emission[:, Observation[t]]

    return F


def backward(Observation, Emission, Transition, Initial):
    """
    Performs the backward algorithm for a hidden markov model.

    Args:
        Observation: numpy.ndarray of shape (T,) containing the index of the
            observation.
        Emission: numpy.ndarray of shape (M, N) containing the emission
            probabilities.
        Transition: numpy.ndarray of shape (M, M) containing the transition
            probabilities.
        Initial: numpy.ndarray of shape (M, 1) containing the starting
            probabilities.

    Returns:
        B, a numpy.ndarray of shape (M, T) containing the backward path
        probabilities.
    """
    T = Observation.shape[0]
    M = Emission.shape[0]
    B = np.zeros((M, T))
    B[:, T - 1] = 1

    for t in range(T - 2, -1, -1):
        B[:, t] = np.matmul(Transition,
                            Emission[:, Observation[t + 1]] * B[:, t + 1])

    return B


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    Performs the Baum-Welch algorithm for a hidden markov model.

    Args:
        Observations: numpy.ndarray of shape (T,) that contains the index of
            the observation. T is the number of observations.
        Transition: numpy.ndarray of shape (M, M) that contains the
            initialized transition probabilities. M is the number of hidden
            states.
        Emission: numpy.ndarray of shape (M, N) that contains the initialized
            emission probabilities. N is the number of output states.
        Initial: numpy.ndarray of shape (M, 1) that contains the initialized
            starting probabilities.
        iterations: number of times expectation-maximization should be
            performed.

    Returns:
        The converged Transition, Emission, or None, None on failure.
    """
    if not isinstance(Observations, np.ndarray):
        return None, None
    if len(Observations.shape) != 1:
        return None, None
    if not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2:
        return None, None
    M, N = Emission.shape
    if not isinstance(Transition, np.ndarray) or Transition.shape != (M, M):
        return None, None
    if not isinstance(Initial, np.ndarray) or Initial.shape != (M, 1):
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    T = Observations.shape[0]
    Transition = Transition.copy()
    Emission = Emission.copy()

    for _ in range(iterations):
        F = forward(Observations, Emission, Transition, Initial)
        B = backward(Observations, Emission, Transition, Initial)

        xi = np.zeros((M, M, T - 1))
        for t in range(T - 1):
            step = (F[:, t, np.newaxis] * Transition *
                    Emission[:, Observations[t + 1]] * B[:, t + 1])
            xi[:, :, t] = step / np.sum(step)

        gamma = np.sum(xi, axis=1)
        Transition = (np.sum(xi, axis=2) /
                      np.sum(gamma, axis=1)[:, np.newaxis])

        gamma = np.hstack((gamma, np.sum(xi[:, :, T - 2],
                                         axis=0)[:, np.newaxis]))
        new_emission = np.zeros((M, N))
        for i in range(N):
            new_emission[:, i] = np.sum(gamma[:, Observations == i], axis=1)
        Emission = new_emission / np.sum(gamma, axis=1)[:, np.newaxis]

    return Transition, Emission
