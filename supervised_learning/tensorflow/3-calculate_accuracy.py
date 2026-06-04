#!/usr/bin/env python3
"""this module calculates the accuracy of a prediction"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """returns a tensor with the accuracy"""
    correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    return tf.reduce_mean(tf.cast(correct, tf.float32))
