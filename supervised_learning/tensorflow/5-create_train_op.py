#!/usr/bin/env python3
"""this module creates the training operation for the network"""
import tensorflow as tf


def create_train_op(loss, alpha):
    """returns the training operation"""
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    return optimizer.minimize(loss)
