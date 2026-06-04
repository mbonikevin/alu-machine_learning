#!/usr/bin/env python3
"""this module calculates the l2 regularized cost of a tensor network"""
import tensorflow as tf


def l2_reg_cost(cost):
    """returns the cost accounting for l2 regularization"""
    return cost + tf.losses.get_regularization_losses()
