import os
import warnings
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from hmmlearn.hmm import GaussianHMM

# Suppress warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)

# --- 1. Configuration ---
TICKER = "QQQ"
PERIOD = "729d"
INTERVAL = "1h"
TRAIN_SPLIT_RATIO = 0.7
VOLATILITY_WINDOW = 24
N_STATES = 2 # COMPRESSION and EXPANSION
OUTPUT_DIR = "hmmlearn_final_results"

# --- 2. Data Preparation ---
def fetch_and_prepare_data(ticker, period, interval, vol_window):
    print(f"Fetching {period} of {interval} data for {ticker}...")
    stock_data = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
    returns = stock_data['Close'].pct_change()
    volatility = returns.rolling(window=vol_window).std().dropna()
    aligned_stock_data = stock_data.loc[volatility.index]
    print(f"Data prepared. Shape of volatility series: {volatility.shape}")
    # hmmlearn requires input of shape (n_samples, n_features)
    return volatility.values.reshape(-1, 1), aligned_stock_data

# --- 3. Main Execution ---
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    volatility, stock_data = fetch_and_prepare_data(TICKER, PERIOD, INTERVAL, VOLATILITY_WINDOW)
    
    split_idx = int(len(volatility) * TRAIN_SPLIT_RATIO)
    train_vol = volatility[:split_idx]
    test_vol = volatility[split_idx:]
    print(f"Data split: Training set size={len(train_vol)}, Test set size={len(test_vol)}")

    # Initialize the HMM model
    # n_iter=100 increases the chance of finding the best fit
    model = GaussianHMM(n_components=N_STATES, covariance_type="full", n_iter=100, random_state=42)

    # Train the model
    print("Fitting HMM model...")
    model.fit(train_vol)
    print("Model fitting complete.")

    # --- Enforce Label Ordering (COMPRESSION < EXPANSION) ---
    # This is our robust, manual replacement for the 'ordered' transform
    low_vol_state = np.argmin(model.means_)
    high_vol_state = np.argmax(model.means_)
    
    print(f"Discovered States - Low Volatility Mean: {model.means_[low_vol_state][0]:.6f}, High Volatility Mean: {model.means_[high_vol_state][0]:.6f}")

    # Predict regimes on train and test data
    print("Predicting regimes...")
    train_regimes_raw = model.predict(train_vol)
    test_regimes_raw = model.predict(test_vol)
    all_regimes_raw = np.concatenate([train_regimes_raw, test_regimes_raw])
    
    # Remap labels to be consistent: 0 = COMPRESSION, 1 = EXPANSION
    all_regimes = np.where(all_regimes_raw == low_vol_state, 0, 1)

    # Save the trained model
    with open(os.path.join(OUTPUT_DIR, "hmm_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    
    # --- Plotting ---
    print("Generating diagnostic plot...")
    fig, ax = plt.subplots(figsize=(15, 7))
    plot_data = stock_data
    ax.plot(plot_data.index, plot_data['Close'], color='black', lw=0.5)
    
    ax.fill_between(plot_data.index, ax.get_ylim()[0], ax.get_ylim()[1], where=all_regimes==0, color='green', alpha=0.3, label="COMPRESSION (Low-Vol)")
    ax.fill_between(plot_data.index, ax.get_ylim()[0], ax.get_ylim()[1], where=all_regimes==1, color='red', alpha=0.3, label="EXPANSION (High-Vol)")
    
    split_time = plot_data.index[split_idx]
    ax.axvline(x=split_time, color='blue', linestyle='--', lw=2, label='Train/Test Split')
    ax.set_title(f"Volatility HMM Regimes for {TICKER} (Hourly) - hmmlearn")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "diagnostic_regimes.png"))
    plt.close()

    print(f"\n✅ HMM analysis complete. Results are in '{OUTPUT_DIR}'.")