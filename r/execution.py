################################################################################
############################## PROGRAM EXECUTION ###############################
################################################################################

import os
import numpy as np
from scipy.stats import chi2
import warnings
warnings.filterwarnings("ignore")
# ============================= OPTIONS =========================================

# running whole program
run_program = True

# printing additional information
add_info = True

# extracting full data files to folder
extract_files = True

# save final result tables
save_results = True

# save test sets
save_test_sets = True

################################################################################

if run_program is True:

    # ------------------------- WORKING DIRECTORY -------------------------------
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(base_dir)
    except Exception:
        pass

    print(os.getcwd())

    # ------------------------- RUN PIPELINE ------------------------------------
    from initialisation import *
    from api_yahoo_finance import download_stock
    from traditional_models import apply_traditional_models
    from GARCH_models import apply_garch_models
    from backtesting import run_backtesting
    

    # --------------------------------------------------------------------------
    # DATA LOADING
    # --------------------------------------------------------------------------
    stock_data = {}

    for s in stocks:
        df = download_stock(s, start, end)
        if df is not None:
            stock_data[s] = df

    # --------------------------------------------------------------------------
    # TRADITIONAL MODELS
    # --------------------------------------------------------------------------
    stock_data = apply_traditional_models(
        stock_data=stock_data,
        stocks=list(stock_data.keys()),
        alpha=alpha,
        d=d
    )

    # --------------------------------------------------------------------------
    # GARCH MODELS
    # --------------------------------------------------------------------------
#   d = 250  # rolling window length (you can change later)

    stock_data, best_models = apply_garch_models(
        stock_data=stock_data,
        stocks=list(stock_data.keys()),
        alpha=alpha,
        d=d,test_start=test_start
    )

    # --------------------------------------------------------------------------
    # BACKTESTING
    # --------------------------------------------------------------------------
    results_backtesting, final_model_summary, test_sets = run_backtesting(
        stock_data=stock_data,
        stocks=list(stock_data.keys()),
        Pf=Pf,
        test_start=test_start,
        test_end=test_end,
        source_NNet_results=source_NNet_results,
        alpha=alpha
    )
    
    # --------------------------------------------------------------------------
    # LOAD NNET RESULTS (if enabled)
    # --------------------------------------------------------------------------
    if source_NNet_results:
        from NNet_sourcer import load_nnet_results
        test_sets = load_nnet_results(
            test_sets=test_sets,
            stocks=list(stock_data.keys()),
            period=period,
            alpha=alpha,
            Pf=Pf,
            n_samples_MC=n_samples_MC,
            NNet_names=NNet_names
        )
        
        # Re-run backtesting with NNet results populated
        print("\n----- Re-running backtesting with LSTM-MDN results -----")
        
        # Update stock_data test sets with NNet results
        for s in stock_data.keys():
            mask = (stock_data[s]["Date"] >= test_start) & (stock_data[s]["Date"] <= test_end)
            if s in test_sets:
                for col in ["LSTM_MDN_vanilla", "LSTM_MDN_reg", "LSTM_MDN_3C"]:
                    if col in test_sets[s].columns:
                        stock_data[s].loc[mask, col] = test_sets[s][col].values
        
        # Re-run backtesting
        results_backtesting, final_model_summary, test_sets = run_backtesting(
            stock_data=stock_data,
            stocks=list(stock_data.keys()),
            Pf=Pf,
            test_start=test_start,
            test_end=test_end,
            source_NNet_results=False,  # Don't reinitialize
            alpha=alpha
        )

    print("Execution successful")
    print(" .\n.\n.\n.\n.")

    # ------------------------- PRINT RESULTS -----------------------------------
    print("----- RESULTS BACKTESTING -----")
    print(" .\n.")

    for res in results_backtesting:
        print(res.round(3))

    print(
        "Chi-sqrd critical value [df = 1, a = 5%]: ",
        chi2.ppf(0.95, df=1)
    )

    print(
        "Chi-sqrd critical value [df = 2, a = 5%]: ",
        chi2.ppf(0.95, df=2)
    )

    ############################################################################
    # CREATE VISUALIZATIONS
    ############################################################################
    try:
        from Visualisations import create_all_visualizations
        create_all_visualizations(
            stock_data=stock_data,
            test_sets=test_sets,
            results_backtesting=results_backtesting,
            stocks=list(stock_data.keys()),
            period=period
        )
    except Exception as e:
        print(f"\n⚠ Warning: Could not create visualizations: {e}")
        print("  (matplotlib may not be installed)")


    ############################################################################
    # SAVING FULL DATA FILES (OPTIONAL)
    ############################################################################
    if extract_files is True:

        extraction_path = os.path.join(os.getcwd(), "data_files")
        os.makedirs(extraction_path, exist_ok=True)

        for s in stock_data:
            stock_data[s].to_csv(
                os.path.join(extraction_path, f"{s}.csv"),
                index=False
            )

    ############################################################################
    # SAVING TEST SETS
    ############################################################################
    if save_test_sets is True:

        folder_test_sets = os.path.join(os.getcwd(), "test_sets")
        os.makedirs(folder_test_sets, exist_ok=True)

        for s in stock_data:

            filename = os.path.join(
                folder_test_sets,
                f"{s}_test_set_{period}.csv"
            )

            test_df = stock_data[s].loc[
                (stock_data[s]["Date"] >= test_start) &
                (stock_data[s]["Date"] <= test_end)
            ]

            test_df.to_csv(filename, index=False)

            print(
                "----- test set for",
                s,
                "(",
                period,
                "period ) successfully saved -----"
            )

    ############################################################################
    # ADDITIONAL INFORMATION
    ############################################################################
    if add_info is True:

        print("----- ADDITIONAL INFORMATION OF TEST RUN -----")
        print(" .")

        print("time span for testing:")
        print(test_start, "until", test_end)

        print(" .")
        print("underlying alpha-level:", alpha)
        print(" .")

        if data_source == "yahoo_finance":
            print("Data from Yahoo Finance")
        else:
            print("Data from Refinitiv Datastream")

        print(" .")

        # distribution info (from GARCH selection)
        if "best_models" in locals():
            for i, s in enumerate(stock_data):
                if "GED" in best_models[i]:
                    print(
                        "Estimated distribution for returns of",
                        s,
                        ": GED"
                    )
                else:
                    print(
                        "Estimated distribution for returns of",
                        s,
                        ": Gaussian"
                    )

################################################################################
# SAVE FINAL SUMMARY TABLE (OPTIONAL)
################################################################################

if save_results is True:
    from save_results import save_comprehensive_summary
    save_comprehensive_summary(
        stocks=list(stock_data.keys()),
        results_backtesting=results_backtesting,
        period=period,
        output_dir="results/final_tables"
    )
    print("Results successfully saved!")
