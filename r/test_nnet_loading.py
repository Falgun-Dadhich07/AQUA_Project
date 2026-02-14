import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import sys
import os

# Ensure we can import from current directory
sys.path.append(os.getcwd())

from NNet_sourcer import load_nnet_results

# Mock data
s = "RELIANCE.NS"
period = "calm"
alpha = 0.99
Pf = 1
n_samples_MC = 100 
NNet_names = ["vanilla", "regularized", "C3"]

# Mock test_sets
df = pd.DataFrame({"Date": pd.date_range("2017-01-01", "2018-12-31")})
test_sets = {s: df}

print("Starting load_nnet_results test...")
updated_sets = load_nnet_results(test_sets, [s], period, alpha, Pf, n_samples_MC, NNet_names)

print("\nResult Analysis:")
print("Columns:", updated_sets[s].columns.tolist())
cols = ["LSTM_MDN_vanilla", "LSTM_MDN_reg", "LSTM_MDN_3C"]
for c in cols:
    if c in updated_sets[s].columns:
        cnt = updated_sets[s][c].count()
        print(f"{c}: {cnt}")
    else:
        print(f"{c}: MISSING")
