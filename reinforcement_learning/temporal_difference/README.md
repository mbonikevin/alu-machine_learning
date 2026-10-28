# Temporal Difference

Model-free prediction and control on `FrozenLake8x8-v0`.

## Tasks

| File | Function | Description |
| --- | --- | --- |
| `0-monte_carlo.py` | `monte_carlo` | First-visit Monte Carlo value estimation |
| `1-td_lambtha.py` | `td_lambtha` | TD(λ) with eligibility traces (backward view) |
| `2-sarsa_lambtha.py` | `sarsa_lambtha` | SARSA(λ) control with ε-greedy and exponential ε decay |

## Requirements

- Ubuntu 16.04 LTS, `python3` (3.5), `numpy` (1.15), `gym` (0.7)
- `pycodestyle` 2.4, all modules/functions documented, all files executable
