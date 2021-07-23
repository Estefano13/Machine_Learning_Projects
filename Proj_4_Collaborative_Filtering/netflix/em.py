"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture
from typing import Tuple
import numpy as np
from common import GaussianMixture
import math
import common
from scipy.special import logsumexp

def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    #array of d because it chages everytime
    d_array = np.count_nonzero(X, axis=1)
    n, _ = X.shape
    K, _ = mixture.mu.shape
    post = np.ones((n, K))

    #Masked array
    X_try = np.ma.masked_equal(X, 0)
    X_try.compressed()

    for i in range(K):
        for j in range(n):
            if d_array[j] > 0:
                post[j, i] = np.log(mixture.p[i] + 1e-16) - (d_array[j] / 2) * (np.log(2 * math.pi) + np.log(mixture.var[i] + 1e-16)) - ((((X_try[j] - mixture.mu[i]) ** 2)).sum()) / (2 * mixture.var[i] + 1e-16)
            else:
                post[j, i] = np.log(mixture.p[i] + 1e-16)
    log_post_sums_K = logsumexp(post, axis=1)

    tiled_vector = np.tile(log_post_sums_K, (K, 1))
    log_post = np.subtract(post, tiled_vector.T)

    posteriors = np.exp(log_post)
    ll = (log_post_sums_K).sum()

    posteriors = np.where(posteriors < np.finfo(float).eps, np.finfo(float).eps, posteriors)
    return posteriors, ll
    raise NotImplementedError




def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """

    n, d_tot = X.shape
    _, K = post.shape

    # Masked array
    X_try = np.ma.masked_equal(X, 0)
    X_try.compressed()

    #binary array of zeros and ones in location
    d_array = np.where(X == 0, 0, 1)
    d_array_sum = d_array.sum(axis=1)

    n_hat_j = post.sum(axis=0)
    p_hat_j = n_hat_j/n

    #new matrices
    mu = np.zeros((K, d_tot))
    var = np.zeros(K)

    #denom = np.zeros((n,K))

    #old matrices
    mu_old = mixture.mu

    for j in range(K):
        mu[j, :] = (post[:, j] @ X) / (np.multiply(post[:, j], d_array.T).sum(axis=1))
        mu[j, :] = np.where(mu[j, :] >= (post[:, j] @ X), mu_old[j,], mu[j,])
        mu_for_var = np.tile(mu[j, :], (n, 1))
        mu_for_var = np.multiply(mu_for_var, d_array)
        nume_var = ((np.subtract(X, mu_for_var)) ** 2).sum(axis=1) @ post[:, j]
        var[j] = nume_var / (post[:, j] * d_array_sum).sum()
        var[j] = np.where(var[j] < min_variance, min_variance, var[j])
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
    ll_new = 0
    ll_old = 0
    while (ll_old == 0 or ll_new - ll_old > abs(ll_new) * 1e-6):
        ll_old = ll_new
        post, ll_new = estep(X, mixture)
        mixture = mstep(X, post, mixture, 0.25)

    return mixture, post, ll_new
    raise NotImplementedError



def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    # array of d because it chages everytime
    d_array = np.count_nonzero(X, axis=1)
    n, _ = X.shape
    K, _ = mixture.mu.shape
    post = np.ones((n, K))

    # Masked array
    X_try = np.ma.masked_equal(X, 0)
    X_try.compressed()

    for i in range(K):
        for j in range(n):
            post[j, i] = np.log(mixture.p[i] + 1e-16) - (d_array[j] / 2) * (np.log(2 * math.pi) + np.log(mixture.var[i] + 1e-16)) - ((((X_try[j] - mixture.mu[i]) ** 2)).sum()) / (2 * mixture.var[i] + 1e-16)

    log_post_sums_K = logsumexp(post, axis=1)

    tiled_vector = np.tile(log_post_sums_K, (K, 1))
    log_post = np.subtract(post, tiled_vector.T)

    posteriors = np.exp(log_post)
    posteriors = np.where(posteriors < np.finfo(float).eps, np.finfo(float).eps, posteriors)

    X_pred = np.where(X==0,posteriors@mixture.mu,X)

    return X_pred
    raise NotImplementedError



#X = np.loadtxt("test_incomplete.txt")
#Mu = np.array([[2.00570178, 4.99062403, 3.13772745, 4.00124767, 1.16193276],
# [2.99396416, 4.68350343, 3.00527213, 3.52422521, 3.08969957],
# [2.54539306, 4.20213487, 4.56501823, 4.55520636, 2.31130827],
# [1.01534912, 4.99975322, 3.49251807, 3.99998124, 4.99986013]])
#Var= np.array([0.25,       0.25,       0.44961685, 0.27930039])
#P= np.array([0.27660973, 0.35431424, 0.26752518, 0.10155086])
#mixture = common.GaussianMixture(Mu, Var, P)

#print(fill_matrix(X, mixture))
#print(np.array([4.00124767, 3.52422521, 4.55520636, 3.99998124]) @ np.array([8.35114583e-01, 1.26066023e-01, 8.03346942e-03, 3.07859243e-02]))
#print(np.array([1.16193276,3.08969957, 2.31130827,4.99986013]) @ np.array([8.35114583e-01, 1.26066023e-01, 8.03346942e-03, 3.07859243e-02]))