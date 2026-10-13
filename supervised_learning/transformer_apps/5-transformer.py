#!/usr/bin/env python3
"""the transformer network, all of it in one file"""

import numpy as np
import tensorflow.compat.v2 as tf


def positional_encoding(max_seq_len, dm):
    """build the positional encoding vectors"""
    encoding = np.zeros((max_seq_len, dm))
    position = np.arange(max_seq_len)[:, np.newaxis]
    index = np.arange(0, dm, 2)
    angles = position / np.power(10000, index / dm)
    encoding[:, 0::2] = np.sin(angles)
    encoding[:, 1::2] = np.cos(angles)
    return encoding


def sdp_attention(Q, K, V, mask=None):
    """work out the scaled dot product attention"""
    scores = tf.matmul(Q, K, transpose_b=True)
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scores = scores / tf.math.sqrt(dk)
    if mask is not None:
        scores += (mask * -1e9)
    weights = tf.nn.softmax(scores, axis=-1)
    return tf.matmul(weights, V), weights


class MultiHeadAttention(tf.keras.layers.Layer):
    """runs attention several times side by side"""

    def __init__(self, dm, h):
        """set up the projections for the queries, keys and values"""
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
        """run every head and glue the results back together"""
        batch = tf.shape(Q)[0]
        q = self.split_heads(self.Wq(Q), batch)
        k = self.split_heads(self.Wk(K), batch)
        v = self.split_heads(self.Wv(V), batch)
        output, weights = sdp_attention(q, k, v, mask)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch, -1, self.dm))
        return self.linear(output), weights


class EncoderBlock(tf.keras.layers.Layer):
    """attention and a feed forward network, each one with a skip"""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """set up the attention, the dense layers and the norms"""
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """push the input through the block"""
        attention, _ = self.mha(x, x, x, mask)
        attention = self.dropout1(attention, training=training)
        out1 = self.layernorm1(x + attention)
        hidden = self.dense_hidden(out1)
        output = self.dense_output(hidden)
        output = self.dropout2(output, training=training)
        return self.layernorm2(out1 + output)


class DecoderBlock(tf.keras.layers.Layer):
    """attention on itself, then on the encoder, then feed forward"""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """set up the two attentions, the dense layers and the norms"""
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
        """push the input through the block"""
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


class Encoder(tf.keras.layers.Layer):
    """a stack of encoder blocks"""

    def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len,
                 drop_rate=0.1):
        """set up the embedding, the positions and the blocks"""
        super(Encoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """embed the input, add the positions and run every block"""
        seq_len = int(x.shape[1])
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += tf.cast(self.positional_encoding[:seq_len], tf.float32)
        x = self.dropout(x, training=training)
        for block in self.blocks:
            x = block(x, training, mask)
        return x


class Decoder(tf.keras.layers.Layer):
    """a stack of decoder blocks"""

    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len,
                 drop_rate=0.1):
        """set up the embedding, the positions and the blocks"""
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
        """embed the target, add the positions and run every block"""
        seq_len = int(x.shape[1])
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += tf.cast(self.positional_encoding[:seq_len], tf.float32)
        x = self.dropout(x, training=training)
        for block in self.blocks:
            x = block(x, encoder_output, training, look_ahead_mask,
                      padding_mask)
        return x


class Transformer(tf.keras.Model):
    """an encoder, a decoder and the layer that picks the words"""

    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """set up the encoder, the decoder and the output layer"""
        super(Transformer, self).__init__()
        self.encoder = Encoder(N, dm, h, hidden, input_vocab,
                               max_seq_input, drop_rate)
        self.decoder = Decoder(N, dm, h, hidden, target_vocab,
                               max_seq_target, drop_rate)
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training, encoder_mask,
             look_ahead_mask, decoder_mask):
        """run the input through the encoder and then the decoder"""
        encoder_output = self.encoder(inputs, training, encoder_mask)
        decoder_output = self.decoder(target, encoder_output, training,
                                      look_ahead_mask, decoder_mask)
        return self.linear(decoder_output)
