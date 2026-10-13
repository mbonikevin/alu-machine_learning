#!/usr/bin/env python3
"""forward propagation for a bidirectional RNN"""

import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """run the cell forwards and backwards over the data

    X has shape (t, m, i)
    h_0 is the first hidden state going forwards, shape (m, h)
    h_t is the last hidden state going backwards, shape (m, h)
    returns the concatenated hidden states and the outputs
    """
    t, m, i = X.shape
    h = h_0.shape[1]
    Hf = np.zeros((t, m, h))
    Hb = np.zeros((t, m, h))
    h_prev = h_0
    for step in range(t):
        h_prev = bi_cell.forward(h_prev, X[step])
        Hf[step] = h_prev
    h_next = h_t
    for step in range(t - 1, -1, -1):
        h_next = bi_cell.backward(h_next, X[step])
        Hb[step] = h_next
    H = np.concatenate((Hf, Hb), axis=2)
    return H, bi_cell.output(H)
