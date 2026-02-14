################################################################################
############ FINAL MODEL (LSTM-MDN-3C) SUMMARY TABLE ############################
################################################################################

import os
import pandas as pd

def save_final_model_summary(
    stocks,
    results_out,
    results_out_pval,
    period,
    output_dir
):
    """
    Exact Python equivalent of the R final summary block.

    stocks            : list of stock names
    results_out       : list of dict-like objects (Kupiec outputs)
    results_out_pval  : list of pandas DataFrames or dict-like objects
    period            : string ("calm" / "covid")
    output_dir        : path to results/final_tables
    """

    final_model_summary = []

    for i in range(len(stocks)):

        final_model_summary.append({
            "Company": stocks[i],
            "Overshoot_Ratio":
                results_out[i]["proportion_overshoots"]["LSTM_MDN_3C"],
            "LRuc":
                results_out_pval[i]["LRuc"]["LSTM_MDN_3C"],
            "LRcci":
                results_out_pval[i]["LRcci"]["LSTM_MDN_3C"],
            "LRcc":
                results_out_pval[i]["LRcc"]["LSTM_MDN_3C"]
        })

    final_model_summary = pd.DataFrame(final_model_summary)

    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir,
        f"NIFTY100_FINAL_MODEL_SUMMARY_{period}.csv"
    )

    final_model_summary.to_csv(output_path, index=False)

    print("----- Final NIFTY100 LSTM-MDN-3C summary table saved -----")

    return final_model_summary

def save_comprehensive_summary(
    stocks,
    results_backtesting,
    period,
    output_dir
):
    """
    Saves a comprehensive summary of all models for all stocks.
    
    stocks              : list of stock symbols
    results_backtesting : list of pandas DataFrames (one per stock)
                         Each DF has index=ModelName, cols=Metrics
    period              : string ("calm" / "covid")
    output_dir          : path to results/final_tables
    """
    
    all_results = []
    
    for i, stock in enumerate(stocks):
        df_res = results_backtesting[i].copy()
        df_res["Company"] = stock
        df_res["Model"] = df_res.index
        
        # Reorder columns to put Company and Model first
        cols = ["Company", "Model"] + [c for c in df_res.columns if c not in ["Company", "Model"]]
        df_res = df_res[cols]
        
        all_results.append(df_res)
        
    full_df = pd.concat(all_results, ignore_index=True)
    
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(
        output_dir,
        f"FULL_BACKTESTING_SUMMARY_{period}.csv"
    )
    
    full_df.to_csv(output_path, index=False)
    print(f"----- Comprehensive backtesting results saved to {output_path} -----")
    
    return full_df
