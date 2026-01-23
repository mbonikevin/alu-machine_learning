#!/usr/bin/env python3
"""
plotting a cubic line graph of x-axis = from 0 to 10
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    """
    the main function to plot the line graph
    """
    y = np.arange(0, 11) ** 3
    x = np.arange(0, 11)

    plt.plot(x, y, color='red')
    plt.xlim(0, 10)
    plt.show()


if __name__ == "__main__":
    main()
