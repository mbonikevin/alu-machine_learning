#!/usr/bin/env python3
"""the whole transformer network"""

import tensorflow as tf
Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.Model):
    """an encoder, a decoder and the layer that picks the words"""

    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """set up the encoder, the decoder and the output layer

        N is the number of blocks in each stack
        dm is the model depth
        h is the number of heads
        hidden is the size of the hidden feed forward layer
        input_vocab is the size of the input vocabulary
        target_vocab is the size of the output vocabulary
        max_seq_input is the longest input the model will see
        max_seq_target is the longest target the model will see
        drop_rate is how much of the output gets dropped
        """
        super(Transformer, self).__init__()
        self.encoder = Encoder(N, dm, h, hidden, input_vocab,
                               max_seq_input, drop_rate)
        self.decoder = Decoder(N, dm, h, hidden, target_vocab,
                               max_seq_target, drop_rate)
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training, encoder_mask,
             look_ahead_mask, decoder_mask):
        """run the input through the encoder and then the decoder

        returns the scores of shape (batch, target_seq_len, target_vocab)
        """
        encoder_output = self.encoder(inputs, training, encoder_mask)
        decoder_output = self.decoder(target, encoder_output, training,
                                      look_ahead_mask, decoder_mask)
        return self.linear(decoder_output)
