# api_yahoo_finance.py

import yfinance as yf
import pandas as pd
from utility import log_returns, discrete_returns

def download_stock(symbol, start, end):
    df = yf.download(symbol, start=start, end=end, progress=False, multi_level_index=False)
    df = df.reset_index()
    print(df.columns)
    print(type(df.columns))
    if df.empty:
        return None

    # df = df[["Close"]]
    df["R"] = discrete_returns(df["Close"].values)
    df["r"] = log_returns(df["Close"].values)
    df = df.dropna()

    return df

