#!/usr/bin/env python3
"""polynomial integral"""


def poly_integral(poly, C=0):
    """calculate the integral of a polynomial"""
    if not isinstance(poly, list) or not isinstance(C, int):
        return None

    result = [C]

    for power in range(len(poly)):
        coef = poly[power] / (power + 1)
        if coef.is_integer():
            coef = int(coef)
        result.append(coef)

    while len(result) > 1 and result[-1] == 0:
        result.pop()

    return result
