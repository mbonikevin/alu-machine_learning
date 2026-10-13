#!/usr/bin/env python3
"""attention for machine translation"""

import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """works out the context vector the decoder needs"""

    def __init__(self, units):
        """set up the three dense layers

        units is the number of hidden units in the alignment model
        """
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """score every encoder state and add them up

        s_prev is the previous decoder hidden state (batch, units)
        hidden_states are the encoder outputs (batch, input_seq_len, units)
        returns the context vector and the attention weights
        """
        s_prev = tf.expand_dims(s_prev, 1)
        score = self.V(tf.nn.tanh(self.W(s_prev) + self.U(hidden_states)))
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(weights * hidden_states, axis=1)
        return context, weights
