#!/usr/bin/env python3
"""Full training with the REINFORCE algorithm"""
import numpy as np
from policy_gradient import policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """
    Implements a full training

    env: initial environment
    nb_episodes: number of episodes used for training
    alpha: the learning rate
    gamma: the discount factor
    show_result: if True, render the environment every 1000 episodes

    Returns: all values of the score (sum of all rewards per episode)
    """
    weight = np.random.rand(*env.observation_space.shape,
                            env.action_space.n)
    scores = []
    for episode in range(nb_episodes):
        state = env.reset()[None, :]
        grads, rewards, done = [], [], False
        while not done:
            if show_result and episode % 1000 == 0:
                env.render()
            action, grad = policy_gradient(state, weight)
            state, reward, done, _ = env.step(action)
            state = state[None, :]
            grads.append(grad)
            rewards.append(reward)
        for t, grad in enumerate(grads):
            G = sum(gamma ** k * r for k, r in enumerate(rewards[t:]))
            weight += alpha * grad * G
        scores.append(sum(rewards))
        print("Episode: {} Score: {}".format(episode, scores[-1]),
              end="\r", flush=False)
    return scores
