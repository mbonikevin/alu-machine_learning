#!/usr/bin/env python3
"""Initializes the Q-table"""
import numpy as np


def q_init(env):
    """
    Initializes the Q-table

    env: the FrozenLakeEnv instance

    Returns: the Q-table as a numpy.ndarray of zeros
    """
    return np.zeros((env.observation_space.n, env.action_space.n))
