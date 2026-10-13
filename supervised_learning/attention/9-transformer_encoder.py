#!/usr/bin/env python3
"""encoder of a transformer"""

import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """a stack of encoder blocks"""

    def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len,
                 drop_rate=0.1):
        """set up the embedding, the positions and the blocks

        N is the number of blocks in the stack
        dm is the model depth
        h is the number of heads
        hidden is the size of the hidden feed forward layer
        input_vocab is the size of the input vocabulary
        max_seq_len is the longest sentence the model will see
        drop_rate is how much of the output gets dropped
        """
        super(Encoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """embed the input, add the positions and run every block

        x holds the word indices of shape (batch, input_seq_len)
        returns the encoder output of shape (batch, input_seq_len, dm)
        """
        seq_len = int(x.shape[1])
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += tf.cast(self.positional_encoding[:seq_len], tf.float32)
        x = self.dropout(x, training=training)
        for block in self.blocks:
            x = block(x, training, mask)
        return x
