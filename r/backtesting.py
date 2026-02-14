################################################################################
######################### BENCHMARKING ##########################################
################################################################################

import numpy as np
import pandas as pd

from utility import (
    Kupiec_Coverage_test,
    Christoffersen_Interval_Forecast_Test
)

################################################################################
######################### VERSION WITH NNET RESULTS #############################
################################################################################

def run_backtesting(
    stock_data,
    stocks,
    Pf,
    test_start,
    test_end,
    source_NNet_results,
    alpha
):
    """
    stock_data: dict
        key   = stock symbol
        value = pandas DataFrame with columns:
                Date, R, r, CMM_VaR, non_par_VaR, GARCH_VaR,
                LSTM_MDN_vanilla, LSTM_MDN_reg, LSTM_MDN_3C
    """

    # Prepare stock_data: calculate one_day_loss and adjust VaR signs
    for s in stocks:
        df = stock_data[s].copy()

        # one_day_loss = -R * Pf  (after r)
        df["one_day_loss"] = -df["R"] * Pf

        # sign conventions (EXACT)
        df["CMM_VaR"] = -df["CMM_VaR"] * Pf
        df["non_par_VaR"] = -df["non_par_VaR"] * Pf
        df["GARCH_VaR"] = -df["GARCH_VaR"] * Pf

        # If NNet results are not sourced (i.e., we're not using them from stock_data),
        # initialize their columns to NaN. Otherwise, assume they are already in stock_data.
        if source_NNet_results is True:
            df["LSTM_MDN_vanilla"] = np.nan
            df["LSTM_MDN_reg"] = np.nan
            df["LSTM_MDN_3C"] = np.nan
        # Ensure NNet columns exist even if not sourced, so they can be selected later
        else:
            for col in ["LSTM_MDN_vanilla", "LSTM_MDN_reg", "LSTM_MDN_3C"]:
                if col not in df.columns:
                    df[col] = np.nan

        stock_data[s] = df

    models = [
        "historical",
        "CMM",
        "GARCH",
        "LSTM_MDN_vanilla",
        "LSTM_MDN_reg",
        "LSTM_MDN_3C"
    ]

    ########################################################################
    # Generating test sets
    ########################################################################

    test_sets = {}

    for s in stocks:
        df = stock_data[s]
        # Dynamically select columns that exist in the DataFrame
        cols_to_select = [
            "Date",
            "one_day_loss",
            "CMM_VaR",
            "non_par_VaR",
            "GARCH_VaR",
            "LSTM_MDN_vanilla",
            "LSTM_MDN_reg",
            "LSTM_MDN_3C"
        ]
        existing_cols = [col for col in cols_to_select if col in df.columns]

        test_sets[s] = df.loc[
            (df["Date"] >= test_start) &
            (df["Date"] <= test_end),
            existing_cols
        ].copy()

    ########################################################################
    # RUN BACKTESTING
    ########################################################################

    results_backtesting = []

    for s in stocks:

        res = pd.DataFrame(
            index=models,
            columns=[
                "proportion_overshoots",
                "LRuc",
                "LRcci",
                "LRcc"
            ],
            dtype=float
        )

        df = test_sets[s]

        # Define models and their corresponding VaR columns
        model_configs = [
            ("historical", "non_par_VaR"),
            ("CMM", "CMM_VaR"),
            ("GARCH", "GARCH_VaR"),
            ("LSTM_MDN_vanilla", "LSTM_MDN_vanilla"),
            ("LSTM_MDN_reg", "LSTM_MDN_reg"),
            ("LSTM_MDN_3C", "LSTM_MDN_3C")
        ]

        for model_name, var_col in model_configs:
            # Only run backtesting if the VaR column exists and is not all NaN
            if var_col in df.columns and not df[var_col].isna().all():
                k = Kupiec_Coverage_test(df["one_day_loss"], df[var_col], alpha)
                c = Christoffersen_Interval_Forecast_Test(
                    df["one_day_loss"], df[var_col]
                )

                res.loc[model_name, "LRuc"] = k["POF"]
                res.loc[model_name, "proportion_overshoots"] = k["proportion_overshoots"]
                res.loc[model_name, "LRcci"] = c["LRcci"]
                res.loc[model_name, "LRcc"] = k["POF"] + c["LRcci"]

        results_backtesting.append(res)

    print("--------- module backtesting - finished ---------")

    ########################################################################
    # FINAL COMPANY-WISE TABLE
    ########################################################################

    final_model_summary = []

    for i, s in enumerate(stocks):
        # Check if "LSTM_MDN_3C" model exists in the results and has non-NaN values
        # Use pd.isna() instead of .isna() because the value might be a numpy scalar
        lstm_value = results_backtesting[i].loc["LSTM_MDN_3C", "proportion_overshoots"]
        
        if "LSTM_MDN_3C" in results_backtesting[i].index and not pd.isna(lstm_value):
            final_model_summary.append({
                "Company": s,
                "Overshoot_Ratio": results_backtesting[i].loc["LSTM_MDN_3C", "proportion_overshoots"],
                "LRuc": results_backtesting[i].loc["LSTM_MDN_3C", "LRuc"],
                "LRcci": results_backtesting[i].loc["LSTM_MDN_3C", "LRcci"],
                "LRcc": results_backtesting[i].loc["LSTM_MDN_3C", "LRcc"]
            })
        else:
            # If the model is not tested or has all NaN results, append NaN
            final_model_summary.append({
                "Company": s,
                "Overshoot_Ratio": np.nan,
                "LRuc": np.nan,
                "LRcci": np.nan,
                "LRcc": np.nan
            })


    final_model_summary = pd.DataFrame(final_model_summary)

    return results_backtesting, final_model_summary, test_sets
