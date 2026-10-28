#!/usr/bin/env python3
"""Epsilon-greedy action selection"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Uses epsilon-greedy to determine the next action

    Q: numpy.ndarray containing the q-table
    state: the current state
    epsilon: the epsilon to use for the calculation

    Returns: the next action index
    """
    p = np.random.uniform()
    if p < epsilon:
        return np.random.randint(0, Q.shape[1])
    return np.argmax(Q[state])
