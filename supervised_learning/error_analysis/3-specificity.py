#!/usr/bin/env python3
"""this module calculates the specificity for each class"""
import numpy as np


def specificity(confusion):
    """returns the specificity of each class"""
    total = np.sum(confusion)
    TP = np.diag(confusion)
    FP = np.sum(confusion, axis=0) - TP
    FN = np.sum(confusion, axis=1) - TP
    TN = total - TP - FP - FN
    return TN / (TN + FP)
