#!/usr/bin/env python3
"""this module creates a learning rate decay operation in tensorflow"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """returns the learning rate decay operation"""
    return tf.train.inverse_time_decay(alpha, global_step, decay_step,
                                       decay_rate, staircase=True)
