#!/usr/bin/env python3
"""RNN decoder for machine translation"""

import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """decodes one word at a time with attention"""

    def __init__(self, vocab, embedding, units, batch):
        """set up the embedding, the GRU and the output layer

        vocab is the size of the output vocabulary
        embedding is the size of the embedding vectors
        units is the number of hidden units in the RNN cell
        batch is the batch size
        """
        super(RNNDecoder, self).__init__()
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units, recurrent_initializer='glorot_uniform',
            return_sequences=True, return_state=True)
        self.F = tf.keras.layers.Dense(vocab)

    def call(self, x, s_prev, hidden_states):
        """take one step of the decoder

        x is the previous word of shape (batch, 1)
        s_prev is the previous decoder hidden state
        hidden_states are the encoder outputs
        returns the word scores and the new decoder hidden state
        """
        attention = SelfAttention(self.units)
        context, _ = attention(s_prev, hidden_states)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context, 1), x], axis=-1)
        outputs, s = self.gru(x, initial_state=s_prev)
        y = self.F(tf.squeeze(outputs, axis=1))
        return y, s
