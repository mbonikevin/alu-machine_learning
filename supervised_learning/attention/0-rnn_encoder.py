#!/usr/bin/env python3
"""RNN encoder for machine translation"""

import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """encodes a sentence for machine translation"""

    def __init__(self, vocab, embedding, units, batch):
        """set up the embedding and the GRU

        vocab is the size of the input vocabulary
        embedding is the size of the embedding vectors
        units is the number of hidden units in the RNN cell
        batch is the batch size
        """
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units, recurrent_initializer='glorot_uniform',
            return_sequences=True, return_state=True)

    def initialize_hidden_state(self):
        """start the hidden state off as zeros"""
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """run the input through the embedding and the GRU

        x holds the word indices of shape (batch, input_seq_len)
        initial is the starting hidden state
        returns the outputs of the encoder and the last hidden state
        """
        x = self.embedding(x)
        outputs, hidden = self.gru(x, initial_state=initial)
        return outputs, hidden
