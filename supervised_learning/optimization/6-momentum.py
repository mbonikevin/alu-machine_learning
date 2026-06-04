#!/usr/bin/env python3
"""this module creates a momentum training operation in tensorflow"""
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """returns the momentum optimization operation"""
    optimizer = tf.train.MomentumOptimizer(alpha, beta1)
    return optimizer.minimize(loss)
