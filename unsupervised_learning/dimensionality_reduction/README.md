# Dimensionality Reduction

Principal components analysis implemented from scratch with `numpy`, using
the singular value decomposition.

## Requirements

* Ubuntu 16.04 LTS, `python3` (3.5)
* `numpy` 1.15
* `pycodestyle` 2.4

## Files

| File | Description |
| --- | --- |
| `0-pca.py` | `pca(X, var=0.95)` — returns the weights matrix `W` of shape `(d, nd)` that keeps `var` fraction of the variance of an already mean-centered `X` |
| `1-pca.py` | `pca(X, ndim)` — centers `X` and returns its transformation `T` of shape `(n, ndim)` |

## Data

The main files are tested with
[`mnist2500_X.txt`](https://lvdmaaten.github.io/tsne/code/tsne_python.zip) and
`mnist2500_labels.txt`, which are not tracked here because of their size.
