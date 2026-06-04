#!/usr/bin/env python3
"""this module calculates the normalization constants of a matrix"""
import numpy as np


def normalization_constants(X):
    """returns the mean and standard deviation of each feature"""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return mean, std
