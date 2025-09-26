import os
import warnings
import yfinance as yf
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import cloudpickle
from scipy import stats

# Suppress warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

# --- 1. Configuration ---
TICKER = "QQQ"
PERIOD = "729d"
INTERVAL = "1h"
TRAIN_SPLIT_RATIO = 0.7
VOLATILITY_WINDOW = 24
N_STATES = 2 # Two states: COMPRESSION and EXPANSION
OUTPUT_DIR = "hmm_native_results"
DRAWS = 2000
TUNE = 2000
CHAINS = 4

# --- 2. Data Preparation ---
def fetch_and_prepare_data(ticker, period, interval, vol_window):
    print(f"Fetching {period} of {interval} data for {ticker}...")
    stock_data = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
    returns = stock_data['Close'].pct_change()
    volatility = returns.rolling(window=vol_window).std().dropna()
    aligned_stock_data = stock_data.loc[volatility.index]
    print(f"Data prepared. Shape of volatility series: {volatility.shape}")
    return volatility, aligned_stock_data

# --- 3. Prediction Function ---
def get_regimes_from_idata(idata, n_states, n_obs):
    """Extracts the most likely regime sequence from the inference data."""
    # We extract the posterior distribution of the latent states
    posterior_states = idata.posterior["states"].stack(sample=("chain", "draw")).values
    
    # For each time step, we find the most common state (the mode)
    # This is the equivalent of the Viterbi path for a Bayesian model
    most_likely_states = stats.mode(posterior_states, axis=1, keepdims=False).mode
    return most_likely_states

# --- 4. Main Execution ---
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    volatility, stock_data = fetch_and_prepare_data(TICKER, PERIOD, INTERVAL, VOLATILITY_WINDOW)
    
    split_idx = int(len(volatility) * TRAIN_SPLIT_RATIO)
    train_vol = volatility.values[:split_idx]
    test_vol = volatility.values[split_idx:]
    print(f"Data split: Training set size={len(train_vol)}, Test set size={len(test_vol)}")

    with pm.Model() as hmm_model:
        # Priors for transition matrix and initial state
        p_transition = pm.Dirichlet("p_transition", a=np.ones((N_STATES, N_STATES)), shape=(N_STATES, N_STATES))
        p_initial = pm.Dirichlet("p_initial", a=np.ones(N_STATES))

        # Priors for regime parameters (volatility of volatility)
        sigma = pm.HalfNormal("sigma", sigma=0.01, shape=N_STATES)

        # Priors for mean volatilities (COMPRESSION < EXPANSION)
        mus = pm.HalfNormal(
            "mus", 
            sigma=0.01, 
            shape=N_STATES, 
            transform=pm.distributions.transforms.ordered,
            initval=np.array([0.002, 0.008]) # Sensible starting values
        )
        
        # Emission distributions for each state
        emission_dists = pm.Normal.dist(mu=mus, sigma=sigma, shape=(len(train_vol), N_STATES))
        
        # The native PyMC HMM likelihood
        # This single line replaces our entire custom `logp_hmm_forward` function
        # It also returns the latent state sequence, which we capture in `states`
        likelihood, states = pm.HiddenMarkovModel(
            "likelihood",
            P=p_transition,
            observed_dist=emission_dists,
            init_dist=pm.Categorical.dist(p=p_initial),
            observed=train_vol,
            shape=len(train_vol)
        )

    with hmm_model:
        print("Starting MCMC sampling on training data...")
        idata = pm.sample(draws=DRAWS, tune=TUNE, chains=CHAINS, cores=1, target_accept=0.9)
    
    print("Saving inference data and model...")
    idata.to_netcdf(os.path.join(OUTPUT_DIR, "hmm_native_inference_data.nc"))
    with open(os.path.join(OUTPUT_DIR, "hmm_native_model.pkl"), "wb") as f: cloudpickle.dump(hmm_model, f)
    
    # Predict regimes on the training set using the inferred latent states
    train_regimes = get_regimes_from_idata(idata, N_STATES, len(train_vol))

    # For the test set, we will use a simplified prediction (as full posterior prediction is complex)
    # We'll use the mean of the learned parameters to find the most likely state
    posterior_data = idata.posterior.stack(sample=("chain", "draw"))
    model_params = {
        'mus': posterior_data['mus'].mean(axis=-1).values,
        'sigmas': posterior_data['sigma'].mean(axis=-1).values,
        'p_regime': posterior_data['p_initial'].mean(axis=-1).values # Use initial as a proxy for stationary dist
    }
    # Simple prediction for test set (similar to mixture model's prediction)
    vol_col = test_vol[:, np.newaxis]
    like_matrix = pm.logp(pm.Normal.dist(mu=model_params['mus'], sigma=model_params['sigmas']), vol_col).eval()
    log_probs = np.log(model_params['p_regime']) + like_matrix
    test_regimes = np.argmax(log_probs, axis=1)

    all_regimes = np.concatenate([train_regimes, test_regimes])
    
    print("Generating diagnostic plots...")
    az.plot_trace(idata, var_names=["p_transition", "sigma", "mus"])
    plt.savefig(os.path.join(OUTPUT_DIR, "diagnostic_trace.png"))
    plt.close()
    
    fig, ax = plt.subplots(figsize=(15, 7))
    plot_data = stock_data
    ax.plot(plot_data.index, plot_data['Close'], color='black', lw=0.5)
    
    ax.fill_between(plot_data.index, ax.get_ylim()[0], ax.get_ylim()[1], where=all_regimes==0, color='green', alpha=0.3, label="COMPRESSION (Low-Vol)")
    ax.fill_between(plot_data.index, ax.get_ylim()[0], ax.get_ylim()[1], where=all_regimes==1, color='red', alpha=0.3, label="EXPANSION (High-Vol)")
    
    split_time = plot_data.index[split_idx]
    ax.axvline(x=split_time, color='blue', linestyle='--', lw=2, label='Train/Test Split')
    ax.set_title(f"Volatility HMM Regimes for {TICKER} (Hourly) - Native PyMC")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "diagnostic_regimes.png"))
    plt.close()

    print(f"\n✅ Walk-forward analysis complete. Results are in '{OUTPUT_DIR}'.")