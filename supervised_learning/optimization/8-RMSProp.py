#!/usr/bin/env python3
"""this module creates an rmsprop training operation in tensorflow"""
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """returns the rmsprop optimization operation"""
    optimizer = tf.train.RMSPropOptimizer(alpha, decay=beta2,
                                          epsilon=epsilon)
    return optimizer.minimize(loss)
