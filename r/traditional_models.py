################################################################################
########## MEAN-VARIANCE MODEL AND HISTORICAL SIMULATIONS #######################
################################################################################

import numpy as np
from scipy.stats import norm

################################################################################
################### CMM MODEL (parametric) ######################################
################################################################################

def par_VaR(returns, alpha, d):
    """
    Exact Python equivalent of:

    par_VaR <- function(returns, alpha, d)
    """

    z = norm.ppf(1 - alpha)

    VaR = np.full(len(returns), np.nan)

    # R loop: for (i in (d + 2):length(returns))
    # Python index correction: i runs from d+1 to len-1
    for i in range(d + 1, len(returns)):

        window = returns[(i - 1):(i - (d + 1)):-1]

        mu = np.nanmean(window)
        sigma = np.nanstd(window, ddof=1)

        VaR[i] = mu + z * sigma

    return VaR


################################################################################
################### HISTORICAL SIMULATIONS (non-parametric) #####################
################################################################################

def non_par_VaR(returns, alpha, d):
    """
    Exact Python equivalent of:

    non_par_VaR <- function(returns, alpha, d)
    """

    VaR = np.full(len(returns), np.nan)

    for i in range(d + 1, len(returns)):

        window = returns[(i - 1):(i - (d + 1)):-1]

        VaR[i] = np.nanquantile(window, 1 - alpha)

    return VaR


################################################################################
################### APPLYING MODELS TO EACH STOCK ###############################
################################################################################

def apply_traditional_models(stock_data, stocks, alpha, d):
    """
    stock_data : dict
        key   = stock symbol
        value = pandas DataFrame with column 'R'

    Exact equivalent of the R for-loop with assign()
    """

    for s in stocks:

        df = stock_data[s]

        cmm = par_VaR(df["R"].values, alpha, d)
        hist = non_par_VaR(df["R"].values, alpha, d)

        # FORCE LENGTH MATCH (EXACT SAME LOGIC)
        if len(cmm) != len(df):
            tmp = np.full(len(df), np.nan)
            tmp[len(df) - len(cmm):] = cmm
            cmm = tmp

        if len(hist) != len(df):
            tmp = np.full(len(df), np.nan)
            tmp[len(df) - len(hist):] = hist
            hist = tmp

        df["CMM_VaR"] = cmm
        df["non_par_VaR"] = hist

        stock_data[s] = df

    print("--------- module traditional_models - finished ---------")

    return stock_data
