#!/usr/bin/env python3
"""
plotting exponential decay of c-14 over
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    """
    the main function to plot the graph
    """
    x = np.arange(0, 28651, 5730)
    r = np.log(0.5)
    t = 5730
    y = np.exp((r / t) * x)

    plt.plot(x, y)
    plt.xlabel("Time (years)")
    plt.ylabel("Fraction Remaining")
    plt.title("Exponential Decay of C-14")
    plt.yscale("log")
    plt.xlim(0, 28650)
    plt.show()


if __name__ == "__main__":
    main()
