#!/usr/bin/env python3
"""this module creates the x and y placeholders for the network"""
import tensorflow as tf


def create_placeholders(nx, classes):
    """returns x and y placeholders"""
    x = tf.placeholder(tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(tf.float32, shape=[None, classes], name='y')
    return x, y
