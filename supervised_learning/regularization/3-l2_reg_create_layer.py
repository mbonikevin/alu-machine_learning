#!/usr/bin/env python3
"""this module creates a tensorflow layer that includes l2 regularization"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """returns the output of the new layer"""
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    reg = tf.contrib.layers.l2_regularizer(lambtha)
    layer = tf.layers.Dense(n, activation=activation,
                            kernel_initializer=init,
                            kernel_regularizer=reg)
    return layer(prev)
