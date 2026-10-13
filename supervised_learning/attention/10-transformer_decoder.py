#!/usr/bin/env python3
"""decoder of a transformer"""

import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """a stack of decoder blocks"""

    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len,
                 drop_rate=0.1):
        """set up the embedding, the positions and the blocks

        N is the number of blocks in the stack
        dm is the model depth
        h is the number of heads
        hidden is the size of the hidden feed forward layer
        target_vocab is the size of the output vocabulary
        max_seq_len is the longest sentence the model will see
        drop_rate is how much of the output gets dropped
        """
        super(Decoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [DecoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask,
             padding_mask):
        """embed the target, add the positions and run every block

        x holds the word indices of shape (batch, target_seq_len)
        returns the decoder output of shape (batch, target_seq_len, dm)
        """
        seq_len = int(x.shape[1])
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += tf.cast(self.positional_encoding[:seq_len], tf.float32)
        x = self.dropout(x, training=training)
        for block in self.blocks:
            x = block(x, encoder_output, training, look_ahead_mask,
                      padding_mask)
        return x
