#!/usr/bin/env python3
"""polynomial derivative"""


def poly_derivative(poly):
    """calculate the derivative of a polynomial"""
    if not isinstance(poly, list) or len(poly) == 0:
        return None

    result = []
    for power in range(1, len(poly)):
        result.append(poly[power] * power)

    if not result:
        return [0]

    return result