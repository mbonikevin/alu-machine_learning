#!/usr/bin/env python3
"""positional encoding for a transformer"""

import numpy as np


def positional_encoding(max_seq_len, dm):
    """build the positional encoding vectors

    max_seq_len is the longest sentence the model will see
    dm is the model depth
    returns a (max_seq_len, dm) array of the encoding vectors
    """
    encoding = np.zeros((max_seq_len, dm))
    position = np.arange(max_seq_len)[:, np.newaxis]
    index = np.arange(0, dm, 2)
    angles = position / np.power(10000, index / dm)
    encoding[:, 0::2] = np.sin(angles)
    encoding[:, 1::2] = np.cos(angles)
    return encoding
