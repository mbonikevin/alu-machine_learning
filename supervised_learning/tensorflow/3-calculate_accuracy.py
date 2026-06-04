#!/usr/bin/env python3
"""accuracy"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """returns a tensor with the accuracy"""
    correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    return tf.reduce_mean(tf.cast(correct, tf.float32))
