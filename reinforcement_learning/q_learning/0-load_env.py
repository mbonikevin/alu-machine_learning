#!/usr/bin/env python3
"""Loads the FrozenLake environment from OpenAI's gym"""
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Loads the pre-made FrozenLakeEnv environment

    desc: None or a list of lists containing a custom description of the map
    map_name: None or a string containing the pre-made map to load
    is_slippery: boolean to determine if the ice is slippery

    If both desc and map_name are None, a random 8x8 map is generated

    Returns: the environment
    """
    return FrozenLakeEnv(desc=desc, map_name=map_name,
                         is_slippery=is_slippery)
