"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture
import math
import common




def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    n, d = X.shape
    K, _ = mixture.mu.shape
    post = np.ones((n, K))

    # normal per row n
    def N_pdf(mu, var, X):
        return (1 / (2 * math.pi * var) ** (d / 2)) * np.exp((-1 / (2 * var)) * (((X - mu) ** 2).sum()))

    for i in range(K):
        for j in range(n):
            post[j, i] = mixture.p[i] * N_pdf(mixture.mu[i], mixture.var[i], X[j])

    post_sums_K = post.sum(axis=1)

    def one(x):
        return 1 / x

    post_sums_K_rev = np.asarray(list(map(one, post_sums_K)))
    tiled_vector = np.tile(post_sums_K_rev, (K, 1))

    # log likelihood

    ll = np.log(post_sums_K).sum()

    return np.multiply(tiled_vector.T, post), ll
    raise NotImplementedError


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    _, K = post.shape

    n_hat_j = post.sum(axis=0)
    p_hat_j = n_hat_j / n

    mu = np.zeros((K, d))
    var = np.zeros(K)

    for j in range(K):
        mu[j, :] = post[:, j] @ X / n_hat_j[j]
        var[j] = (((X - mu[j]) ** 2).sum(axis=1) @ post[:, j]) / (n_hat_j[j] * d)
    return common.GaussianMixture(mu, var, p_hat_j)
    raise NotImplementedError


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    ll_new = None
    ll_old = None
    while(ll_old is None or ll_new - ll_old > abs(ll_new)*1e-6):
        ll_old = ll_new
        post, ll_new = estep(X, mixture)
        mixture = mstep(X, post)

    return mixture, post, ll_new
    raise NotImplementedError
