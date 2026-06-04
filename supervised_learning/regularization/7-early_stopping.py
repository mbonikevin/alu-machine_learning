#!/usr/bin/env python3
"""this module determines if gradient descent should stop early"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """returns whether to stop early and the updated count"""
    if opt_cost - cost > threshold:
        count = 0
    else:
        count += 1
    if count >= patience:
        return True, count
    return False, count
