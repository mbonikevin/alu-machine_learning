# Hidden Markov Models

Markov chains and hidden markov models with `numpy`: state probabilities
after n steps, steady states for regular chains, absorbing chain detection,
and the forward, Viterbi, backward and Baum-Welch algorithms.

## Requirements

* Ubuntu 16.04 LTS, `python3` (3.5)
* `numpy` 1.15
* `pycodestyle` 2.4

## Files

| File | Description |
| --- | --- |
| `0-markov_chain.py` | `markov_chain(P, s, t=1)` — probability of being in each state after `t` iterations |
| `1-regular.py` | `regular(P)` — steady state probabilities of a regular markov chain |
| `2-absorbing.py` | `absorbing(P)` — whether a markov chain is absorbing |
| `3-forward.py` | `forward(Observation, Emission, Transition, Initial)` — forward algorithm, returns the likelihood and the forward path probabilities |
| `4-viterbi.py` | `viterbi(Observation, Emission, Transition, Initial)` — most likely sequence of hidden states |
| `5-backward.py` | `backward(Observation, Emission, Transition, Initial)` — backward algorithm, returns the likelihood and the backward path probabilities |
| `6-baum_welch.py` | `baum_welch(Observations, Transition, Emission, Initial, iterations=1000)` — fits the transition and emission probabilities |
