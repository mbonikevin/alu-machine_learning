#!/usr/bin/env python3
"""this module updates the learning rate using inverse time decay"""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """returns the updated value for alpha in a stepwise fashion"""
    return alpha / (1 + decay_rate * (global_step // decay_step))
