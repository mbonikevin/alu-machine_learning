#!/usr/bin/env python3
"""this module performs bayesian optimization on a gaussian process"""
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """performs bayesian optimization on a noiseless 1d gaussian process"""

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1,
                 sigma_f=1, xsi=0.01, minimize=True):
        """sets the attributes for the optimization"""
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1],
                               ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """returns the next best sample point and the expected improvement"""
        mu, sigma = self.gp.predict(self.X_s)
        if self.minimize:
            opt = np.min(self.gp.Y)
            imp = opt - mu - self.xsi
        else:
            opt = np.max(self.gp.Y)
            imp = mu - opt - self.xsi
        with np.errstate(divide='ignore', invalid='ignore'):
            Z = imp / sigma
            EI = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            EI[sigma == 0] = 0
        X_next = self.X_s[np.argmax(EI)]
        return X_next, EI

    def optimize(self, iterations=100):
        """returns the optimal point and function value"""
        for i in range(iterations):
            X_next, _ = self.acquisition()
            if X_next in self.gp.X:
                break
            Y_next = self.f(X_next)
            self.gp.update(X_next, Y_next)
        if self.minimize:
            idx = np.argmin(self.gp.Y)
        else:
            idx = np.argmax(self.gp.Y)
        self.gp.X = self.gp.X[:-1]
        X_opt = self.gp.X[idx]
        Y_opt = self.gp.Y[idx]
        return X_opt, Y_opt
