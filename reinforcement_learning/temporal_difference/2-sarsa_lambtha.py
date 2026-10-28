#!/usr/bin/env python3
"""SARSA(lambtha) algorithm with eligibility traces"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Uses epsilon-greedy to determine the next action

    Q: numpy.ndarray containing the q-table
    state: the current state
    epsilon: the epsilon to use for the calculation

    Returns: the next action index
    """
    if np.random.uniform(0, 1) > epsilon:
        return np.argmax(Q[state])
    return np.random.randint(0, Q.shape[1])


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs SARSA(lambtha)

    env: the openAI environment instance
    Q: numpy.ndarray of shape (s, a) containing the Q table
    lambtha: the eligibility trace factor
    episodes: total number of episodes to train over
    max_steps: maximum number of steps per episode
    alpha: the learning rate
    gamma: the discount rate
    epsilon: the initial threshold for epsilon greedy
    min_epsilon: the minimum value that epsilon should decay to
    epsilon_decay: the decay rate for updating epsilon between episodes

    Returns: Q, the updated Q table
    """
    init_epsilon = epsilon

    for episode in range(episodes):
        state = env.reset()
        action = epsilon_greedy(Q, state, epsilon)
        eligibility = np.zeros(Q.shape)

        for step in range(max_steps):
            new_state, reward, done, _ = env.step(action)
            new_action = epsilon_greedy(Q, new_state, epsilon)

            delta = reward + gamma * Q[new_state, new_action] \
                - Q[state, action]

            eligibility *= gamma * lambtha
            eligibility[state, action] += 1.0

            Q += alpha * delta * eligibility

            if done:
                break
            state = new_state
            action = new_action

        epsilon = min_epsilon + (init_epsilon - min_epsilon) * \
            np.exp(-epsilon_decay * episode)

    return Q
