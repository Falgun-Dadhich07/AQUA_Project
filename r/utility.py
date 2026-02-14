################################################################################
############################## Utility script ##################################
################################################################################

import numpy as np
from scipy.stats import chi2

################################################################################
############################# UTILITY FUNCTIONS #################################
################################################################################

#########################
# Return functions
#########################

def log_returns(price):
    """
    Exact Python equivalent of:
    log_returns <- function(price)
    """
    r = np.full(len(price), np.nan)

    for i in range(1, len(r)):
        r[i] = np.log(price[i]) - np.log(price[i - 1])

    return r


def discrete_returns(price):
    """
    Exact Python equivalent of:
    discrete_returns <- function(price)
    """
    R = np.full(len(price), np.nan)

    for i in range(1, len(R)):
        R[i] = (price[i] - price[i - 1]) / price[i - 1]

    return R


################################################################################
####################### Backtesting utility ####################################
################################################################################

def Kupiec_Coverage_test(Losses, VaR, a):
    """
    Exact Python equivalent of:
    Kupiec_Coverage_test <- function(Losses, VaR, a = alpha)
    """

    # Indicator <- ifelse(Losses < VaR, 0, 1)
    Indicator = np.where(Losses < VaR, 0, 1)

    I_alpha = Indicator.sum()
    sample_T = len(Indicator)

    alpha_exp = 1 - a
    alpha_hat = (1 / sample_T) * I_alpha

    A = (1 - alpha_hat) / (1 - alpha_exp)
    B = alpha_hat / alpha_exp

    POF = 2 * np.log((A ** (sample_T - I_alpha)) * (B ** I_alpha))

    p_val = 1 - chi2.cdf(POF, df=1)

    expected_overshoots = alpha_exp * sample_T
    proportion_overshoots = alpha_hat

    return {
        "expected_overshoots": expected_overshoots,
        "n_overshoots": I_alpha,
        "proportion_overshoots": proportion_overshoots,
        "sample": sample_T,
        "POF": POF,
        "p_val": p_val
    }


def Christoffersen_Interval_Forecast_Test(Losses, VaR):
    """
    Exact Python equivalent of:
    Christoffersen_Interval_Forecast_Test <- function(Losses, VaR)
    """

    Indicator = np.where(Losses < VaR, 0, 1)

    n_z_z = 0
    n_z_o = 0
    n_o_z = 0
    n_o_o = 0

    for i in range(len(Indicator) - 1):
        if Indicator[i] == 0 and Indicator[i + 1] == 0:
            n_z_z += 1
        elif Indicator[i] == 0 and Indicator[i + 1] == 1:
            n_z_o += 1
        elif Indicator[i] == 1 and Indicator[i + 1] == 0:
            n_o_z += 1
        elif Indicator[i] == 1 and Indicator[i + 1] == 1:
            n_o_o += 1

    pi_zero = n_z_o / (n_z_z + n_z_o) if (n_z_z + n_z_o) > 0 else 0
    pi_one  = n_o_o / (n_o_z + n_o_o) if (n_o_z + n_o_o) > 0 else 0
    pi      = (n_z_o + n_o_o) / (n_z_z + n_z_o + n_o_z + n_o_o)

    numerator = ((1 - pi) ** (n_z_z + n_o_z)) * (pi ** (n_z_o + n_o_o))
    denominator = ((1 - pi_zero) ** n_z_z) * (pi_zero ** n_z_o) * \
                  ((1 - pi_one) ** n_o_z) * (pi_one ** n_o_o)

    LRcci = -2 * np.log(numerator / denominator) if denominator > 0 else np.nan
    p_val = 1 - chi2.cdf(LRcci, df=1)

    return {
        "LRcci": LRcci,
        "p_val": p_val
    }


################################################################################
############# UTILITY FUNCTIONS FOR GAUSSIAN MIXTURE SAMPLING ##################
################################################################################

def r_gaussmix_2c(N, mu1, mu2, sigma1, sigma2, pi1, pi2):
    """
    Exact Python equivalent of:
    r_gaussmix_2c <- function(N, mu1, mu2, sigma1, sigma2, pi1, pi2)
    """

    if pi1 + pi2 < 0.99999:
        raise ValueError("Error")

    U = np.random.rand(N)
    samples = np.full(N, np.nan)

    for i in range(N):
        if U[i] < pi1:
            samples[i] = np.random.normal(mu1, sigma1)
        else:
            samples[i] = np.random.normal(mu2, sigma2)

    return samples


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
        raise ValueError("Error")

    U = np.random.rand(N)
    samples = np.full(N, np.nan)

    for i in range(N):
        if U[i] < pi1:
            samples[i] = np.random.normal(mu1, sigma1)
        elif U[i] < (pi1 + pi2):
            samples[i] = np.random.normal(mu2, sigma2)
        else:
            samples[i] = np.random.normal(mu3, sigma3)

    return samples
