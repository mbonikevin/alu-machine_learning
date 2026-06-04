#!/usr/bin/env python3
"""this module creates a confusion matrix"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """returns the confusion matrix of shape (classes, classes)"""
    return np.matmul(labels.T, logits)
