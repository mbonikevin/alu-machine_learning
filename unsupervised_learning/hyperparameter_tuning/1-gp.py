#!/usr/bin/env python3
"""this module represents a noiseless 1d gaussian process"""
import numpy as np


class GaussianProcess:
    """represents a noiseless 1d gaussian process"""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """sets the attributes and the covariance kernel matrix"""
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """returns the rbf covariance kernel matrix between two matrices"""
        sqdist = np.sum(X1 ** 2, 1).reshape(-1, 1) + \
            np.sum(X2 ** 2, 1) - 2 * np.matmul(X1, X2.T)
        return self.sigma_f ** 2 * np.exp(-0.5 / self.l ** 2 * sqdist)

    def predict(self, X_s):
        """returns the mean and variance of points in the process"""
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)
        mu = np.matmul(np.matmul(K_s.T, K_inv), self.Y).reshape(-1)
        cov = K_ss - np.matmul(np.matmul(K_s.T, K_inv), K_s)
        sigma = np.diag(cov)
        return mu, sigma
