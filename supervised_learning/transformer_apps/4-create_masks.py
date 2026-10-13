#!/usr/bin/env python3
"""masks for training a transformer"""

import tensorflow.compat.v2 as tf


def create_masks(inputs, target):
    """build the three masks the transformer needs

    inputs holds the input sentences of shape (batch, seq_len_in)
    target holds the target sentences of shape (batch, seq_len_out)
    returns the encoder mask, the combined mask for the first decoder
    attention and the decoder mask for the second one
    """
    seq_len_out = target.shape[1]
    encoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    encoder_mask = encoder_mask[:, tf.newaxis, tf.newaxis, :]
    decoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    decoder_mask = decoder_mask[:, tf.newaxis, tf.newaxis, :]
    look_ahead = 1 - tf.linalg.band_part(
        tf.ones((seq_len_out, seq_len_out)), -1, 0)
    target_mask = tf.cast(tf.math.equal(target, 0), tf.float32)
    target_mask = target_mask[:, tf.newaxis, tf.newaxis, :]
    combined_mask = tf.maximum(target_mask, look_ahead)
    return encoder_mask, combined_mask, decoder_mask
