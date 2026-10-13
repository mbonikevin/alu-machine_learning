#!/usr/bin/env python3
"""scaled dot product attention"""

import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """work out the scaled dot product attention

    Q holds the query matrix, K the key matrix and V the value matrix
    mask, when it is given, is broadcast onto the scores and hides
    whatever it marks
    returns the attention output and the attention weights
    """
    scores = tf.matmul(Q, K, transpose_b=True)
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scores = scores / tf.math.sqrt(dk)
    if mask is not None:
        scores += (mask * -1e9)
    weights = tf.nn.softmax(scores, axis=-1)
    return tf.matmul(weights, V), weights
