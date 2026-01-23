#!/usr/bin/env python3
"""
plotting a histogram of student grades for project A
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    """
    the main function to plot the histogram
    """
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)

    bins = np.arange(0, 101, 10)

    plt.hist(student_grades, bins=bins, edgecolor='black')
    plt.xlabel("Grades")
    plt.ylabel("Number of Students")
    plt.title("Project A")
    plt.show()


if __name__ == "__main__":
    main()
