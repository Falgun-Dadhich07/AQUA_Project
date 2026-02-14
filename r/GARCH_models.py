################################################################################
########################### GARCH MODELS #######################################
################################################################################

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import norm
from scipy.stats import gennorm  # GED equivalent

################################################################################
#################### VaR FORECAST FROM GARCH MODEL ##############################
################################################################################

def GARCH_VaR(model_res, alpha_lvl):
    """
    Exact equivalent of:
    GARCH_VaR <- function(model, alpha_lvl = alpha)
    """

    dist = model_res.model.distribution.name

    # ===== choose quantile =====
    if dist.lower() == "ged":
        shape = model_res.params["nu"]
        quantile = gennorm.ppf(1 - alpha_lvl, beta=shape)
    else:  # normal
        quantile = norm.ppf(1 - alpha_lvl)

    # last conditional volatility (sigma_t)
    sigma_t = model_res.conditional_volatility[-1]

    # last residual (epsilon_t)
    epsilon_t = model_res.resid[-1]

    # parameters
    omega = model_res.params["omega"]
    alpha1 = model_res.params["alpha[1]"]
    beta1 = model_res.params["beta[1]"]

    # sigma^2_{t+1}
    sigma_sq_t1 = omega + alpha1 * epsilon_t**2 + beta1 * sigma_t**2
    sigma_t1 = np.sqrt(sigma_sq_t1)

    VaR_forecast = quantile * sigma_t1

    return VaR_forecast


################################################################################
#################### FITTING GARCH(1,1) MODELS ##################################
################################################################################

def apply_garch_models(stock_data, stocks, alpha, d,test_start):
    """
    Exact semantic equivalent of GARCH_models.R
    """

    GARCH_1_1_normal = []
    GARCH_1_1_GED = []

    # ===== initial fits (full sample, for AIC comparison) =====
    for s in stocks:

        returns = stock_data[s]["R"].dropna().values

        model_norm = arch_model(
            returns,
            vol="GARCH",
            p=1,
            q=1,
            mean="Zero",
            dist="normal"
        ).fit(disp="off")

        model_ged = arch_model(
            returns,
            vol="GARCH",
            p=1,
            q=1,
            mean="Zero",
            dist="ged"
        ).fit(disp="off")

        GARCH_1_1_normal.append(model_norm)
        GARCH_1_1_GED.append(model_ged)

    ################################################################################
    #################### AUTOMATED MODEL SELECTION (AIC) ##########################
    ################################################################################

    best_models = []

    for i in range(len(stocks)):
        if GARCH_1_1_normal[i].aic < GARCH_1_1_GED[i].aic:
            best_models.append("normal")
        else:
            best_models.append("ged")

    print("optimal models from automated model selection (AIC):")
    print(best_models)

    ################################################################################
    ###################### ROLLING VaR FORECASTS #################################
    ################################################################################

    for i, s in enumerate(stocks):

        df = stock_data[s].copy()
        df = df.reset_index(drop=True)  # Reset index for integer-based indexing

        # index where test period starts
        test_mask = df["Date"] >= test_start
        if test_mask.sum() == 0:
            print(f"Warning: No data found for {s} after {test_start}")
            df["GARCH_VaR"] = np.nan
            stock_data[s] = df
            continue
            
        t_plus_one = df[test_mask].index[0]

        storer = np.full(len(df), np.nan)

        while True:

            t = t_plus_one - 1
            s_idx = max(0, t - d)  # Ensure we don't go negative

            returns = df.loc[s_idx:t, "R"].values

            # Skip if not enough data
            if len(returns) < 10:
                t_plus_one += 1
                if t_plus_one >= len(df):
                    break
                continue

            # refit model on rolling window
            try:
                working_model = arch_model(
                    returns,
                    vol="GARCH",
                    p=1,
                    q=1,
                    mean="Zero",
                    dist=best_models[i]
                ).fit(disp="off")

                storer[t_plus_one] = GARCH_VaR(
                    model_res=working_model,
                    alpha_lvl=alpha
                )
            except Exception as e:
                print(f"Warning: GARCH fit failed at index {t_plus_one} for {s}: {e}")

            if t_plus_one >= len(df) - 1:
                break

            t_plus_one += 1

        df["GARCH_VaR"] = storer
        stock_data[s] = df

    print("--------- module GARCH_models - finished ---------")

    return stock_data, best_models
