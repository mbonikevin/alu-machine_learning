#!/usr/bin/env python3
"""forward propagation for a deep RNN"""

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """run a stack of cells over every time step of the data

    rnn_cells is the list of cells, one per layer
    X has shape (t, m, i) and h_0 has shape (l, m, h)
    returns all of the hidden states and all of the outputs
    """
    t, m, i = X.shape
    layers = len(rnn_cells)
    h = h_0.shape[2]
    H = np.zeros((t + 1, layers, m, h))
    H[0] = h_0
    Y = []
    for step in range(t):
        x = X[step]
        for layer in range(layers):
            h_next, y = rnn_cells[layer].forward(H[step, layer], x)
            H[step + 1, layer] = h_next
            x = h_next
        Y.append(y)
    return H, np.array(Y)
