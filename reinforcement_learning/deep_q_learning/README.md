# Deep Q-learning

A DQN agent that plays Atari's Breakout, built with `keras`, `keras-rl` and `gym`.

## Files

| File | Description |
| --- | --- |
| `train.py` | Trains a `DQNAgent` with `SequentialMemory` and an annealed `EpsGreedyQPolicy`, saves the policy network to `policy.h5` |
| `play.py` | Loads `policy.h5` and plays 10 visualized episodes with a `GreedyQPolicy` |

## Notes

- Frames are preprocessed to 84x84 grayscale `uint8` and stacked 4 deep.
- Training runs 5M steps with 50k warmup steps; epsilon anneals 1.0 -> 0.1 over 1M steps.
- Requires `gym` with the Atari ROMs installed (`pip install gym[atari]`), `keras` 2.2.x and `keras-rl`.
