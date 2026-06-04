#!/usr/bin/env python3
"""this module normalizes a matrix using mean and standard deviation"""


def normalize(X, m, s):
    """returns the normalized X matrix"""
    return (X - m) / s
