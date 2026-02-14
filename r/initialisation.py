# initialisation.py

import pandas as pd
from pathlib import Path

# ---------------- PERIOD ----------------
period = "covid"   # "calm" or "covid"

# ---------------- DATA SOURCE ------------
data_source = "yahoo_finance"

# ---------------- STOCK UNIVERSE ---------
BASE_DIR = Path(__file__).resolve().parent.parent  # ON NIFTY100
DATA_DIR = BASE_DIR / "data"

stocks = pd.read_csv(DATA_DIR / "nifty100_constituents.csv")["Symbol"][25:51].tolist() #You can change number of symbols by changing 50

# ---------------- TEST PERIOD ------------
if period == "calm":
    test_start = "2017-01-01"
    test_end   = "2018-12-31"
else:
    test_start = "2021-01-01"
    test_end   = "2022-12-31"

# ---------------- MODELS -----------------
models = [
    "historical",
    "CMM",
    "GARCH",
    "LSTM_MDN_vanilla",
    "LSTM_MDN_reg",
    "LSTM_MDN_3C"
]

# ---------------- VAR PARAMETERS ---------
alpha = 0.99
d = 250
Pf = 1

# ---------------- DATA RANGE -------------
start = "2001-01-01"
end   = "2023-05-10"

# ---------------- NNET -------------------
source_NNet_results = True
NNet_names = ["vanilla", "regularized", "C3"]
n_samples_MC = 100000

print(f"Initialisation complete: {len(stocks)} stock(s) loaded for {period} period")
