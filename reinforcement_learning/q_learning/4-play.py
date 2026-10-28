#!/usr/bin/env python3
"""Has a trained agent play an episode"""
import numpy as np


def play(env, Q, max_steps=100):
    """
    Has the trained agent play an episode, always exploiting the Q-table

    env: the FrozenLakeEnv instance
    Q: numpy.ndarray containing the Q-table
    max_steps: maximum number of steps in the episode

    Each state of the board is displayed via the console

    Returns: the total rewards for the episode
    """
    state = env.reset()
    env.render()
    total_rewards = 0

    for step in range(max_steps):
        action = np.argmax(Q[state])
        state, reward, done, _ = env.step(action)
        env.render()
        total_rewards += reward

        if done:
            break

    return total_rewards
