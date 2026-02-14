################################################################################
################## IMPORTING RESULTS FROM LSTM-MDN BACKEND ######################
################################################################################

# NOTE:
# THIS SCRIPT IS SOLELY FOR IMPORTING NEURAL NET RESULTS AS PART OF A UNIVERSITY
# PROJECT
#
# EXACT LOGIC PORT FROM NNet_sourcer.R (NO CHANGES)

import os
import numpy as np
import pandas as pd

from Gaussian_mixture_sampling import (
    r_gaussmix_2c,
    r_gaussmix_3c
)

################################################################################
# FUNCTION TO LOAD AND PROCESS NNET RESULTS
################################################################################

def load_nnet_results(test_sets, stocks, period, alpha, Pf, n_samples_MC, NNet_names):
    """
    Load neural network MDN results and populate test_sets with VaR predictions.
    
    Parameters:
    -----------
    test_sets : dict
        Dictionary of test set DataFrames (one per stock)
    stocks : list
        List of stock symbols
    period : str
        "calm" or "covid"
    alpha : float
        Confidence level (e.g., 0.99)
    Pf : float
        Portfolio value
    n_samples_MC : int
        Number of Monte Carlo samples
    NNet_names : list
        List of neural network model names
        
    Returns:
    --------
    test_sets : dict
        Updated test_sets with LSTM-MDN columns populated
    """
    
    import_names = stocks  # same as R

    NNet_file_names = []
    for x in import_names:
        for y in NNet_names:
            for z in [period]:
                string = f"{x}_{y}_{z}"
                NNet_file_names.append(string)

    # names for result object generation
    NNet_results = [f"{name}_VaR" for name in NNet_file_names]

    ############################################################################
    # IMPORTING MDN OUTPUT CSV FILES
    ############################################################################

    MDN_data = {}

    for name in NNet_file_names:
        # Check both relative locations for robustness
        path1 = os.path.join("NNet_data", f"{name}.csv")
        path2 = os.path.join("..", "NNet_data", f"{name}.csv")
        
        if os.path.exists(path2):
            path = path2
        elif os.path.exists(path1):
            path = path1
        else:
            print(f"Warning: {name}.csv not found in NNet_data or ../NNet_data, skipping...")
            continue
            
        df = pd.read_csv(path)
        # df = df.iloc[:, 1:]  # Commented out: My Python generation does not include index column
        MDN_data[name] = df

    ############################################################################
    # INITIALISING EMPTY VECTORS
    ############################################################################

    VaR_results = {}
    for res in NNet_results:
        VaR_results[res] = []

    ############################################################################
    # MONTE-CARLO SAMPLING FROM GAUSSIAN MIXTURES
    ############################################################################

    for j, res_name in enumerate(NNet_results):

        file_name = NNet_file_names[j]
        
        if file_name not in MDN_data:
            print(f"Skipping {file_name} - data not loaded")
            continue
            
        df = MDN_data[file_name]

        # ===================== 2-COMPONENT MIXTURE =====================
        if "C3" not in res_name:

            for i in range(len(df)):

                sims = r_gaussmix_2c(
                    N      = n_samples_MC,
                    mu1    = df.loc[i, "mu1"],
                    mu2    = df.loc[i, "mu2"],
                    sigma1 = abs(df.loc[i, "sigma1"]),
                    sigma2 = abs(df.loc[i, "sigma2"]),
                    pi1    = df.loc[i, "pi1"],
                    pi2    = df.loc[i, "pi2"]
                )

                VaR = np.quantile(sims, 1 - alpha) * -Pf
                VaR_results[res_name].append(VaR)

            print(f"MC-simulation for {file_name} (2-component): successful")

        # ===================== 3-COMPONENT MIXTURE =====================
        else:

            for i in range(len(df)):

                sims = r_gaussmix_3c(
                    N      = n_samples_MC,
                    mu1    = df.loc[i, "mu1"],
                    mu2    = df.loc[i, "mu2"],
                    mu3    = df.loc[i, "mu3"],
                    sigma1 = abs(df.loc[i, "sigma1"]),
                    sigma2 = abs(df.loc[i, "sigma2"]),
                    sigma3 = abs(df.loc[i, "sigma3"]),
                    pi1    = df.loc[i, "pi1"],
                    pi2    = df.loc[i, "pi2"],
                    pi3    = df.loc[i, "pi3"]
                )

                VaR = np.quantile(sims, 1 - alpha) * -Pf
                VaR_results[res_name].append(VaR)

            print(f"MC-simulation for {file_name} (3-component): successful")

    ############################################################################
    # ASSIGNING RESULTS TO TEST SETS (EXACT SEMANTICS)
    ############################################################################

    for s in stocks:
        test_len = len(test_sets[s])
        
        # vanilla
        var_key = f"{s}_vanilla_{period}_VaR"
        if var_key in VaR_results:
            var_array = np.array(VaR_results[var_key])
            if len(var_array) >= test_len:
                test_sets[s]["LSTM_MDN_vanilla"] = var_array[:test_len]
            else:
                print(f"Warning: Not enough {var_key} predictions ({len(var_array)} < {test_len})")

        # regularized
        var_key = f"{s}_regularized_{period}_VaR"
        if var_key in VaR_results:
            var_array = np.array(VaR_results[var_key])
            if len(var_array) >= test_len:
                test_sets[s]["LSTM_MDN_reg"] = var_array[:test_len]
            else:
                print(f"Warning: Not enough {var_key} predictions ({len(var_array)} < {test_len})")

        # 3-component (FINAL MODEL)
        var_key = f"{s}_C3_{period}_VaR"
        if var_key in VaR_results:
            var_array = np.array(VaR_results[var_key])
            if len(var_array) >= test_len:
                test_sets[s]["LSTM_MDN_3C"] = var_array[:test_len]
                print(f"✓ Assigned {len(var_array[:test_len])} LSTM-MDN predictions for {s}")
            else:
                print(f"Warning: Not enough {var_key} predictions ({len(var_array)} < {test_len})")
    
    print("--------- NNet results loaded successfully ---------")
    
    return test_sets
