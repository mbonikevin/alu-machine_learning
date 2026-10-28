#!/usr/bin/env python3
"""Displays a game of Atari's Breakout played by the trained agent"""
import gym

from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import GreedyQPolicy

train = __import__('train')
AtariProcessor = train.AtariProcessor
build_model = train.build_model
WINDOW_LENGTH = train.WINDOW_LENGTH


if __name__ == '__main__':
    env = gym.make('Breakout-v0')
    env.reset()
    num_actions = env.action_space.n

    model = build_model(num_actions)

    memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
    dqn = DQNAgent(model=model, nb_actions=num_actions,
                   policy=GreedyQPolicy(), memory=memory,
                   processor=AtariProcessor())
    dqn.compile(optimizer='adam', metrics=['mae'])
    dqn.load_weights('policy.h5')

    dqn.test(env, nb_episodes=10, visualize=True)
