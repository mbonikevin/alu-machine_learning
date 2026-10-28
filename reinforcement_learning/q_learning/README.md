# Q-learning

Model-free control on `FrozenLake` with a tabular Q-table.

## Tasks

| File | Function | Description |
| --- | --- | --- |
| `0-load_env.py` | `load_frozen_lake` | Loads the FrozenLakeEnv environment |
| `1-q_init.py` | `q_init` | Initializes the Q-table with zeros |
| `2-epsilon_greedy.py` | `epsilon_greedy` | Epsilon-greedy action selection |
| `3-q_learning.py` | `train` | Q-learning with exponential epsilon decay |
| `4-play.py` | `play` | Plays an episode with the trained agent |

## Requirements

- Ubuntu 16.04 LTS, `python3` (3.5), `numpy` (1.15), `gym` (0.7)
- `pycodestyle` 2.4, all modules/functions documented, all files executable
