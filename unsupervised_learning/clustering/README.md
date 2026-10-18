# Clustering

Unsupervised clustering from scratch with `numpy`, plus the `sklearn` and
`scipy` equivalents: K-means, Gaussian Mixture Models fit by Expectation
Maximization, model selection by variance (elbow) and BIC, and agglomerative
clustering.

## Requirements

* Ubuntu 16.04 LTS, `python3` (3.5)
* `numpy` 1.15, `sklearn` 0.21, `scipy` 1.3
* `pycodestyle` 2.4

## Files

| File | Description |
| --- | --- |
| `0-initialize.py` | `initialize(X, k)` — initializes K-means centroids with a multivariate uniform distribution |
| `1-kmeans.py` | `kmeans(X, k, iterations=1000)` — performs K-means on a dataset |
| `2-variance.py` | `variance(X, C)` — total intra-cluster variance for a data set |
| `3-optimum.py` | `optimum_k(X, kmin=1, kmax=None, iterations=1000)` — tests for the optimum number of clusters by variance |
| `4-initialize.py` | `initialize(X, k)` — initializes the priors, means and covariances of a GMM |
| `5-pdf.py` | `pdf(X, m, S)` — probability density function of a Gaussian distribution |
| `6-expectation.py` | `expectation(X, pi, m, S)` — expectation step of the EM algorithm for a GMM |
| `7-maximization.py` | `maximization(X, g)` — maximization step of the EM algorithm for a GMM |
| `8-EM.py` | `expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False)` — runs EM for a GMM |
| `9-BIC.py` | `BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False)` — best number of clusters by Bayesian Information Criterion |
| `10-kmeans.py` | `kmeans(X, k)` — K-means with `sklearn.cluster` |
| `11-gmm.py` | `gmm(X, k)` — GMM with `sklearn.mixture` |
| `12-agglomerative.py` | `agglomerative(X, dist)` — Ward-linkage agglomerative clustering with `scipy`, plots the dendrogram |
