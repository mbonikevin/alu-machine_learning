#!/usr/bin/env python3
"""this module creates a layer of a neural network using dropout"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """returns the output of the new layer"""
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(n, activation=activation,
                            kernel_initializer=init)
    dropout = tf.layers.Dropout(keep_prob)
    return dropout(layer(prev))
