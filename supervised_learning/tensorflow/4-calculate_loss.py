#!/usr/bin/env python3
"""loss"""
import tensorflow as tf


def calculate_loss(y, y_pred):
    """returns a tensor with the loss"""
    return tf.losses.softmax_cross_entropy(y, y_pred)
