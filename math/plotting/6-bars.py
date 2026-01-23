#!/usr/bin/env python3
"""
plotting a stacked bar graph of fruit quantities per person
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    """
    the main function to plot the stacked bar chart
    """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))

    people = ['Farrah', 'Fred', 'Felicia']
    x = np.arange(len(people))

    apples = fruit[0]
    bananas = fruit[1]
    oranges = fruit[2]
    peaches = fruit[3]

    plt.bar(x, apples, width=0.5, color='red', label='Apples')
    plt.bar(x, bananas, width=0.5, bottom=apples,
            color='yellow', label='Bananas')
    plt.bar(x, oranges, width=0.5, bottom=apples + bananas,
            color='#ff8000', label='Oranges')
    plt.bar(x, peaches, width=0.5, bottom=apples + bananas + oranges,
            color='#ffe5b4', label='Peaches')

    plt.xticks(x, people)
    plt.ylabel("Quantity of Fruit")
    plt.ylim(0, 80)
    plt.yticks(np.arange(0, 81, 10))
    plt.title("Number of Fruit per Person")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
