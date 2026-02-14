###############################################################
# Gaussian Mixture Monte Carlo Sampling
# EXACT PORT OF utility.R + Gaussian_mixture_sampling.R
###############################################################

import numpy as np

###############################################################
# 2-COMPONENT GAUSSIAN MIXTURE
###############################################################

def r_gaussmix_2c(N, mu1, mu2, sigma1, sigma2, pi1, pi2):
    """
    Exact Python equivalent of:

    r_gaussmix_2c <- function(N, mu1, mu2, sigma1, sigma2, pi1, pi2)
    """

    if pi1 + pi2 < 0.99999:
        raise ValueError("Error: mixture weights do not sum to 1")

    U = np.random.rand(N)
    samples = np.zeros(N)

    for i in range(N):
        if U[i] < pi1:
            samples[i] = np.random.normal(mu1, sigma1)
        else:
            samples[i] = np.random.normal(mu2, sigma2)

    return samples


###############################################################
# 3-COMPONENT GAUSSIAN MIXTURE
###############################################################

def r_gaussmix_3c(
    N,
    mu1, mu2, mu3,
    sigma1, sigma2, sigma3,
    pi1, pi2, pi3
):
    """
    Exact Python equivalent of:

    r_gaussmix_3c <- function(
        N, mu1, mu2, mu3,
        sigma1, sigma2, sigma3,
        pi1, pi2, pi3
    )
    """

    if pi1 + pi2 + pi3 < 0.99999:
        raise ValueError("Error: mixture weights do not sum to 1")

    U = np.random.rand(N)
    samples = np.zeros(N)

    for i in range(N):
        if U[i] < pi1:
            samples[i] = np.random.normal(mu1, sigma1)
        elif U[i] < (pi1 + pi2):
            samples[i] = np.random.normal(mu2, sigma2)
        else:
            samples[i] = np.random.normal(mu3, sigma3)

    return samples
