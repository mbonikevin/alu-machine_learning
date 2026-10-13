#!/usr/bin/env python3
"""forward propagation for a simple RNN"""

import numpy as np


def rnn(rnn_cell, X, h_0):
    """run the cell over every time step of the data

    rnn_cell is the instance used to do the steps
    X has shape (t, m, i) and h_0 has shape (m, h)
    returns all of the hidden states and all of the outputs
    """
    t, m, i = X.shape
    h = h_0.shape[1]
    H = np.zeros((t + 1, m, h))
    H[0] = h_0
    Y = []
    for step in range(t):
        h_next, y = rnn_cell.forward(H[step], X[step])
        H[step + 1] = h_next
        Y.append(y)
    return H, np.array(Y)
