#!/usr/bin/env python3
"""Trains a DQN agent to play Atari's Breakout"""
import gym
import numpy as np
from PIL import Image

import keras as K
from keras.layers import Activation, Conv2D, Dense, Flatten, Permute
from keras.models import Sequential
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.core import Processor
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy

INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4


class AtariProcessor(Processor):
    """Downsamples, grayscales and rescales the Atari frames"""

    def process_observation(self, observation):
        """Converts a frame to an 84x84 grayscale uint8 image"""
        assert observation.ndim == 3
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE).convert('L')
        processed = np.array(img).astype('uint8')
        assert processed.shape == INPUT_SHAPE
        return processed

    def process_state_batch(self, batch):
        """Rescales a batch of states to [0, 1]"""
        return batch.astype('float32') / 255.

    def process_reward(self, reward):
        """Clips the reward to [-1, 1]"""
        return np.clip(reward, -1., 1.)


def build_model(num_actions):
    """Builds the convolutional policy network"""
    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=(WINDOW_LENGTH,) + INPUT_SHAPE))
    model.add(Conv2D(32, (8, 8), strides=(4, 4)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(num_actions))
    model.add(Activation('linear'))
    return model


def build_agent(model, num_actions):
    """Builds the DQN agent with its memory and policy"""
    memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps',
                                  value_max=1., value_min=.1, value_test=.05,
                                  nb_steps=1000000)
    dqn = DQNAgent(model=model, nb_actions=num_actions, policy=policy,
                   memory=memory, processor=AtariProcessor(),
                   nb_steps_warmup=50000, gamma=.99,
                   target_model_update=10000, train_interval=4,
                   delta_clip=1.)
    dqn.compile(Adam(lr=.00025), metrics=['mae'])
    return dqn


if __name__ == '__main__':
    env = gym.make('Breakout-v0')
    env.reset()
    num_actions = env.action_space.n

    model = build_model(num_actions)
    model.summary()

    dqn = build_agent(model, num_actions)
    dqn.fit(env, nb_steps=5000000, log_interval=10000, visualize=False,
            verbose=2)
    dqn.save_weights('policy.h5', overwrite=True)
