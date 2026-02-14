# -*- coding: utf-8 -*-
"""
LSTM–MDN for Value-at-Risk forecasting
UPDATED: Compatible with TensorFlow 2.x (Pure TF implementation, no TFP dependency)
"""

import numpy as np
import pandas as pd
import random as rn
import os
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping

# ===============================
# REPRODUCIBILITY
# ===============================

os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
np.random.seed(6969)
rn.seed(6969)
tf.random.set_seed(6969)

# ===============================
# PARAMETERS
# ===============================

d = 10
test_start_date = "2021-01-01"
test_end_date = "2022-12-31"
period = "covid"

save_results = True

mon = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

# ===============================
# LOAD NIFTY100 STOCK LIST
# ===============================

# Get parent directory
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

stocks = pd.read_csv(DATA_DIR / "nifty100_constituents.csv")["Symbol"][:50].tolist()
print(f"Training LSTM-MDN for {len(stocks)} stock(s): {stocks}")

# ===============================
# UTILITIES
# ===============================

def df_to_X_y(df, window_size=10):
    """Convert time series to supervised learning format"""
    df_as_np = df.to_numpy()
    X, y = [], []
    for i in range(len(df_as_np) - window_size):
        X.append([[a] for a in df_as_np[i:i+window_size]])
        y.append(df_as_np[i + window_size])
    return np.array(X), np.array(y)

# ===============================
# MDN LAYER (Built-in, no external library)
# ===============================

class MDNLayer(keras.layers.Layer):
    """Mixture Density Network output layer"""
    
    def __init__(self, output_dim, num_mixtures, **kwargs):
        super(MDNLayer, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.num_mixtures = num_mixtures
        self.mdn_mus = Dense(num_mixtures * output_dim, name='mdn_mus')
        self.mdn_sigmas = Dense(num_mixtures * output_dim, name='mdn_sigmas')
        self.mdn_pi = Dense(num_mixtures, activation='softmax', name='mdn_pi')
        
    def call(self, x):
        mus = self.mdn_mus(x)
        # Use softplus for positive std dev + epsilon for stability
        sigmas = tf.math.softplus(self.mdn_sigmas(x)) + 1e-5
        pi = self.mdn_pi(x)
        return tf.concat([mus, sigmas, pi], axis=1)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "num_mixtures": self.num_mixtures
        })
        return config

# ===============================
# MDN LOSS FUNCTIONS (PURE TENSORFLOW)
# ===============================

def log_normal_pdf(y, mu, sigma):
    """Compute log PDF of normal distribution"""
    # y shape: (batch, 1, output_dim)
    # mu, sigma shape: (batch, num_mixes, output_dim)
    
    # log(1 / (sqrt(2*pi)*sigma)) = -0.5*log(2*pi) - log(sigma)
    # log(exp(...)) = -0.5 * ((y - mu) / sigma)^2
    
    log_2pi = tf.math.log(2.0 * 3.141592653589793)
    log_prop = -0.5 * log_2pi - tf.math.log(sigma) - 0.5 * tf.square((y - mu) / sigma)
    return log_prop

def get_mixture_loss_func(output_dim, num_mixes):
    """Standard MDN loss function"""
    def mdn_loss_func(y_true, y_pred):
        # Split y_pred -> mu, sigma, pi
        # y_pred shape: (batch, num_params)
        out_mu = y_pred[:, :num_mixes * output_dim]
        out_sigma = y_pred[:, num_mixes * output_dim:2 * num_mixes * output_dim]
        out_pi = y_pred[:, 2 * num_mixes * output_dim:]
        
        # Reshape to (batch, num_mixes, output_dim)
        out_mu = tf.reshape(out_mu, [-1, num_mixes, output_dim])
        out_sigma = tf.reshape(out_sigma, [-1, num_mixes, output_dim])
        
        # y_true shape: (batch, output_dim) -> Expand to (batch, 1, output_dim)
        y_true_expanded = tf.reshape(y_true, [-1, 1, output_dim])
        
        # Calculate log probability for each component
        # Result shape: (batch, num_mixes, output_dim)
        log_probs_component = log_normal_pdf(y_true_expanded, out_mu, out_sigma)
        
        # Sum over output dimension (assuming diagonal independence)
        # Result shape: (batch, num_mixes)
        log_probs = tf.reduce_sum(log_probs_component, axis=2)
        
        # Combine with mixture weights
        # log(sum(pi * N(...))) = log(sum(exp(log(pi) + log_prob)))
        # = ReduceLogSumExp(log(pi) + log_prob)
        weighted_log_probs = tf.math.log(out_pi + 1e-8) + log_probs
        
        # Log likelihood of the mixture
        log_likelihood = tf.reduce_logsumexp(weighted_log_probs, axis=1)
        
        # Loss is negative log likelihood
        return -tf.reduce_mean(log_likelihood)
    
    return mdn_loss_func

def get_mixture_loss_func_REGULARIZED(output_dim, num_mixes, lambda_reg=0.1):
    """Regularized MDN loss function"""
    def mdn_loss_func(y_true, y_pred):
        # Split y_pred -> mu, sigma, pi
        out_mu = y_pred[:, :num_mixes * output_dim]
        out_sigma = y_pred[:, num_mixes * output_dim:2 * num_mixes * output_dim]
        out_pi = y_pred[:, 2 * num_mixes * output_dim:]
        
        # Reshape
        out_mu = tf.reshape(out_mu, [-1, num_mixes, output_dim])
        out_sigma = tf.reshape(out_sigma, [-1, num_mixes, output_dim])
        
        # y_true reshaped
        y_true_expanded = tf.reshape(y_true, [-1, 1, output_dim])
        
        # Log probs
        log_probs_component = log_normal_pdf(y_true_expanded, out_mu, out_sigma)
        log_probs = tf.reduce_sum(log_probs_component, axis=2)
        
        # Log likelihood
        weighted_log_probs = tf.math.log(out_pi + 1e-8) + log_probs
        log_likelihood = tf.reduce_logsumexp(weighted_log_probs, axis=1)
        
        # Loss with regularization on pi
        loss = -tf.reduce_mean(log_likelihood)
        loss += lambda_reg * tf.reduce_sum(tf.square(out_pi))
        return loss
    
    return mdn_loss_func

# ===============================
# MDN OUTPUT PARSING
# ===============================

def get_outputs(predictions, output_dim=1, N_MIXES=2):
    """Extract mu, sigma, pi from MDN output"""
    mus = predictions[:, :N_MIXES * output_dim].reshape(-1, N_MIXES, output_dim)
    sigs = predictions[:, N_MIXES * output_dim:2 * N_MIXES * output_dim].reshape(-1, N_MIXES, output_dim)
    pis = predictions[:, 2 * N_MIXES * output_dim:]
    
    # Softmax for pis (already applied in layer, but good safety)
    pis = pis / pis.sum(axis=1, keepdims=True)
    
    return mus.squeeze(), sigs.squeeze(), pis

def MDN_predict(model, test_data, output_dim=1, N_MIXES=2):
    """Make predictions and extract parameters"""
    preds = model.predict(test_data, verbose=0)
    mu, sigma, pi = get_outputs(preds, output_dim, N_MIXES)
    return {"mu": mu, "sigma": sigma, "pi": pi}

def dataframe_converter(pred, N_MIXES=2):
    """Convert predictions to DataFrame"""
    cols = []
    for i in range(1, N_MIXES + 1): 
        cols.append(f"mu{i}")
    for i in range(1, N_MIXES + 1): 
        cols.append(f"sigma{i}")
    for i in range(1, N_MIXES + 1): 
        cols.append(f"pi{i}")
    
    # Ensure dimensions match
    mu_flat = pred["mu"]
    sigma_flat = pred["sigma"]
    
    # If 1D result (e.g. 1 sample), reshape
    if mu_flat.ndim == 1 and N_MIXES > 1:
        mu_flat = mu_flat.reshape(-1, N_MIXES)
        sigma_flat = sigma_flat.reshape(-1, N_MIXES)
        
    return pd.DataFrame(
        np.column_stack([mu_flat, sigma_flat, pred["pi"]]),
        columns=cols
    )

# ===============================
# MAIN LOOP — PER STOCK
# ===============================

# Create output directory
output_dir = BASE_DIR / "NNet_data"
output_dir.mkdir(exist_ok=True)

for stock in stocks:
    print(f"\n{'='*60}")
    print(f"Training MDN models for {stock}")
    print(f"{'='*60}")
    
    # Try to load from r/data_files first (if execution.py was run)
    data_file = BASE_DIR / "r" / "data_files" / f"{stock}.csv"
    
    if data_file.exists():
        print(f"Loading data from: {data_file}")
        df = pd.read_csv(data_file)
        df["Date"] = pd.to_datetime(df["Date"])
    else:
        print(f"Data file not found. Please run r/execution.py first to download stock data.")
        continue
    
    # Filter data by date for training/test split
    # Test data starts from test_start_date
    df = df.sort_values("Date").reset_index(drop=True)
    
    # Find index where test starts
    test_mask = df["Date"] >= test_start_date
    if not test_mask.any():
        print(f"No data found after {test_start_date}. Skipping {stock}.")
        continue
        
    test_start_idx = df[test_mask].index[0]
    
    # Ensure there is enough data for windowing
    if test_start_idx < d:
        print(f"Not enough training data before {test_start_date}. Skipping {stock}.")
        continue
    
    # Define start index for test (accounting for window lookback)
    test_idx_with_lookback = test_start_idx - d
    
    # Limit test data to test_end_date
    test_end_mask = df["Date"] <= test_end_date
    if not test_end_mask.any():
        print(f"No data found before {test_end_date}. Skipping {stock}.")
        continue
        
    # Get the last index for test
    test_end_idx = df[test_end_mask].index[-1]
    
    train_data = df.iloc[:test_start_idx]
    test_data = df.iloc[test_idx_with_lookback : test_end_idx + 1]
    
    print(f"Data Date Range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"Test Period: {test_start_date} to {test_end_date}")
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data) - d} (raw: {len(test_data)})")
    
    if len(train_data) < 100:
        print("Warning: Very small training set.")
    
    X_train, y_train = df_to_X_y(train_data["R"], d)
    X_test, y_test = df_to_X_y(test_data["R"], d)
    
    # Input shape: (batch, d, 1)
    input_shape = (d, 1)
    
    # ===============================
    # MODEL 1 — 2 MIX (VANILLA)
    # ===============================
    
    print("\n[1/3] Training Vanilla 2-component model...")
    model_plain = Sequential([
        InputLayer(input_shape=input_shape),
        LSTM(6, activation="relu"),
        Dense(10, activation="relu"),
        MDNLayer(1, 2)
    ])
    
    model_plain.compile(
        loss=get_mixture_loss_func(1, 2),
        optimizer=keras.optimizers.Adam(learning_rate=0.001)
    )
    
    model_plain.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        callbacks=[mon],
        verbose=0
    )
    
    # Save predictions
    pred_vanilla = MDN_predict(model_plain, X_test, 1, 2)
    df_vanilla = dataframe_converter(pred_vanilla, 2)
    df_vanilla.to_csv(output_dir / f"{stock}_vanilla_{period}.csv", index=False)
    print(f"✓ Saved: {stock}_vanilla_{period}.csv ({len(df_vanilla)} rows)")
    
    # ===============================
    # MODEL 2 — 2 MIX REGULARIZED
    # ===============================
    
    print("\n[2/3] Training Regularized 2-component model...")
    model_reg = Sequential([
        InputLayer(input_shape=input_shape),
        LSTM(6, activation="relu"),
        Dense(10, activation="relu"),
        MDNLayer(1, 2)
    ])
    
    model_reg.compile(
        loss=get_mixture_loss_func_REGULARIZED(1, 2, 0.1),
        optimizer=keras.optimizers.Adam(learning_rate=0.001)
    )
    
    model_reg.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        callbacks=[mon],
        verbose=0
    )
    
    # Save predictions
    pred_reg = MDN_predict(model_reg, X_test, 1, 2)
    df_reg = dataframe_converter(pred_reg, 2)
    df_reg.to_csv(output_dir / f"{stock}_regularized_{period}.csv", index=False)
    print(f"✓ Saved: {stock}_regularized_{period}.csv ({len(df_reg)} rows)")
    
    # ===============================
    # MODEL 3 — 3 MIX (FINAL MODEL)
    # ===============================
    
    print("\n[3/3] Training 3-component model (FINAL)...")
    model_C3 = Sequential([
        InputLayer(input_shape=input_shape),
        LSTM(6, activation="relu"),
        Dense(10, activation="relu"),
        MDNLayer(1, 3)
    ])
    
    model_C3.compile(
        loss=get_mixture_loss_func(1, 3),
        optimizer=keras.optimizers.Adam(learning_rate=0.001)
    )
    
    model_C3.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        callbacks=[mon],
        verbose=0
    )
    
    # Save predictions
    pred_C3 = MDN_predict(model_C3, X_test, 1, 3)
    df_C3 = dataframe_converter(pred_C3, 3)
    df_C3.to_csv(output_dir / f"{stock}_C3_{period}.csv", index=False)
    print(f"✓ Saved: {stock}_C3_{period}.csv ({len(df_C3)} rows)")
    
    # Clear session to free memory
    tf.keras.backend.clear_session()
    
    print(f"\n✓ Completed training for {stock}")

print(f"\n{'='*60}")
print(f"✓ All LSTM-MDN models trained successfully!")
print(f"✓ Results saved to: {output_dir}")
print(f"{'='*60}")
print(f"\nNext step: Run 'python r/execution.py' to include LSTM-MDN in analysis")
