#!/usr/bin/env python3
"""multi head attention"""

import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """runs attention several times side by side"""

    def __init__(self, dm, h):
        """set up the projections for the queries, keys and values

        dm is the model depth and has to divide evenly by h
        h is the number of heads
        """
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch):
        """cut the last axis into heads and put the heads up front"""
        x = tf.reshape(x, (batch, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """run every head and glue the results back together

        returns the attention output and the attention weights
        """
        batch = tf.shape(Q)[0]
        q = self.split_heads(self.Wq(Q), batch)
        k = self.split_heads(self.Wk(K), batch)
        v = self.split_heads(self.Wv(V), batch)
        output, weights = sdp_attention(q, k, v, mask)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch, -1, self.dm))
        return self.linear(output), weights
