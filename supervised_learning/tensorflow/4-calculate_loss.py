#!/usr/bin/env python3
"""this module calculates the softmax cross-entropy loss of a prediction"""
import tensorflow as tf


def calculate_loss(y, y_pred):
    """returns a tensor with the loss"""
    return tf.losses.softmax_cross_entropy(y, y_pred)
