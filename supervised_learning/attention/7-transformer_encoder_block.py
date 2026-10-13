#!/usr/bin/env python3
"""encoder block of a transformer"""

import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """attention and a feed forward network, each one with a skip"""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """set up the attention, the dense layers and the norms

        dm is the model depth
        h is the number of heads
        hidden is the size of the hidden feed forward layer
        drop_rate is how much of the output gets dropped
        """
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """push the input through the block

        x is the input of shape (batch, input_seq_len, dm)
        training says whether the model is training
        mask hides whatever should not be attended to
        returns the block output of shape (batch, input_seq_len, dm)
        """
        attention, _ = self.mha(x, x, x, mask)
        attention = self.dropout1(attention, training=training)
        out1 = self.layernorm1(x + attention)
        hidden = self.dense_hidden(out1)
        output = self.dense_output(hidden)
        output = self.dropout2(output, training=training)
        return self.layernorm2(out1 + output)
