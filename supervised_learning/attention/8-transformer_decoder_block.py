#!/usr/bin/env python3
"""decoder block of a transformer"""

import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """attention on itself, then on the encoder, then feed forward"""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """set up the two attentions, the dense layers and the norms

        dm is the model depth
        h is the number of heads
        hidden is the size of the hidden feed forward layer
        drop_rate is how much of the output gets dropped
        """
        super(DecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask,
             padding_mask):
        """push the input through the block

        x is the input of shape (batch, target_seq_len, dm)
        encoder_output is what the encoder produced
        look_ahead_mask hides the words that come later
        padding_mask hides the padding of the encoder output
        returns the block output of shape (batch, target_seq_len, dm)
        """
        attention1, _ = self.mha1(x, x, x, look_ahead_mask)
        attention1 = self.dropout1(attention1, training=training)
        out1 = self.layernorm1(x + attention1)
        attention2, _ = self.mha2(out1, encoder_output, encoder_output,
                                  padding_mask)
        attention2 = self.dropout2(attention2, training=training)
        out2 = self.layernorm2(out1 + attention2)
        hidden = self.dense_hidden(out2)
        output = self.dense_output(hidden)
        output = self.dropout3(output, training=training)
        return self.layernorm3(out2 + output)
