#!/usr/bin/env python3
"""train op"""
import tensorflow as tf


def create_train_op(loss, alpha):
    """returns the training operation"""
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    return optimizer.minimize(loss)
