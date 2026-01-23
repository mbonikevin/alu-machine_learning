#!/usr/bin/env python3
"""
plotting a scatter graph of men's height vs weight
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    """
    the main function to plot the graph
    """
    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x, y = np.random.multivariate_normal(mean, cov, 2000).T
    y += 180

    plt.scatter(x, y, color='#cd31ca')
    plt.xlabel("Height (in)")
    plt.ylabel("Weight (lbs)")
    plt.title("Men's Height vs Weight")
    plt.show()


if __name__ == "__main__":
    main()
