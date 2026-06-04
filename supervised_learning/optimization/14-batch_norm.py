#!/usr/bin/env python3
"""this module creates a batch normalization layer in tensorflow"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """returns a tensor of the activated output for the layer"""
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, kernel_initializer=init)
    z = layer(prev)
    gamma = tf.Variable(tf.ones([1, n]), name='gamma')
    beta = tf.Variable(tf.zeros([1, n]), name='beta')
    mean, var = tf.nn.moments(z, axes=[0])
    z_norm = tf.nn.batch_normalization(z, mean, var, beta, gamma, 1e-8)
    return activation(z_norm)
