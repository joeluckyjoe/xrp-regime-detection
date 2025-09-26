import os
import warnings
import yfinance as yf
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import pytensor.tensor as pt
import matplotlib.pyplot as plt
import cloudpickle

# Suppress warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

# --- 1. Configuration ---
TICKER = "QQQ"
PERIOD = "729d"
INTERVAL = "1h"
TRAIN_SPLIT_RATIO = 0.7
OUTPUT_DIR = "static_model_results"
DRAWS = 2000
TUNE = 2000
CHAINS = 4

# --- 2. Data Preparation ---
def fetch_data(ticker, period, interval):
    print(f"Fetching {period} of {interval} data for {ticker}...")
    stock_data = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
    returns = stock_data['Close'].pct_change().dropna()
    print(f"Data fetched. Shape of returns: {returns.shape}")
    return returns, stock_data

# --- 3. Prediction Function ---
def get_regime_probabilities(returns, model_params):
    mus = model_params['mus']
    sigmas = model_params['sigmas']
    p_regime = model_params['p_regime']
    
    returns_col = returns[:, np.newaxis]
    
    like_matrix = pm.logp(pm.Normal.dist(mu=mus, sigma=sigmas), returns_col).eval()
    
    log_probs = np.log(p_regime) + like_matrix
    
    return np.argmax(log_probs, axis=1)

# --- 4. Main Execution ---
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    returns, stock_data = fetch_data(TICKER, PERIOD, INTERVAL)
    split_idx = int(len(returns) * TRAIN_SPLIT_RATIO)
    train_returns = returns.values[:split_idx]
    test_returns = returns.values[split_idx:]
    print(f"Data split: Training set size={len(train_returns)}, Test set size={len(test_returns)}")

    with pm.Model() as static_model:
        p_regime = pm.Dirichlet("p_regime", a=np.ones(3))
        sigma = pm.HalfNormal("sigma", sigma=0.01, shape=3)
        mus = pm.Normal(
            "mus", 
            mu=0, 
            sigma=0.01, 
            shape=3, 
            transform=pm.distributions.transforms.ordered,
            initval=np.array([-0.001, 0.0, 0.001])
        )
        
        comp_dists = pm.Normal.dist(mu=mus, sigma=sigma)
        
        likelihood = pm.Mixture(
            "likelihood",
            w=p_regime,
            comp_dists=comp_dists,
            observed=train_returns[:, None]
        )

    with static_model:
        print("Starting MCMC sampling on training data...")
        idata = pm.sample(draws=DRAWS, tune=TUNE, chains=CHAINS, cores=1, target_accept=0.9)
    
    print("Saving inference data and model...")
    idata.to_netcdf(os.path.join(OUTPUT_DIR, "static_inference_data.nc"))
    with open(os.path.join(OUTPUT_DIR, "static_model.pkl"), "wb") as f: cloudpickle.dump(static_model, f)
    
    posterior_data = idata.posterior.stack(sample=("chain", "draw"))
    model_params = {
        'p_regime': posterior_data['p_regime'].mean(axis=-1).values,
        'sigmas': posterior_data['sigma'].mean(axis=-1).values,
        'mus': posterior_data['mus'].mean(axis=-1).values
    }

    print("Predicting regimes on train and test data...")
    train_regimes = get_regime_probabilities(train_returns, model_params)
    test_regimes = get_regime_probabilities(test_returns, model_params)
    all_regimes = np.concatenate([train_regimes, test_regimes])
    
    print("Generating diagnostic plots...")
    az.plot_trace(idata, var_names=["p_regime", "sigma", "mus"])
    plt.savefig(os.path.join(OUTPUT_DIR, "diagnostic_trace.png"))
    plt.close()
    
    fig, ax = plt.subplots(figsize=(15, 7))
    plot_data = stock_data.iloc[1:]
    ax.plot(plot_data.index, plot_data['Close'], color='black', lw=0.5)

    # --- FIX: Add parentheses to ax.get_ylim() in the following three lines ---
    ax.fill_between(plot_data.index, ax.get_ylim()[0], ax.get_ylim()[1], where=all_regimes==0, color='red', alpha=0.3, label="BEAR")
    ax.fill_between(plot_data.index, ax.get_ylim()[0], ax.get_ylim()[1], where=all_regimes==1, color='gray', alpha=0.3, label="SIDEWAYS")
    ax.fill_between(plot_data.index, ax.get_ylim()[0], ax.get_ylim()[1], where=all_regimes==2, color='green', alpha=0.3, label="BULL")

    split_time = plot_data.index[split_idx]
    ax.axvline(x=split_time, color='blue', linestyle='--', lw=2, label='Train/Test Split')
    ax.set_title(f"Walk-Forward Market Regimes for {TICKER} (Hourly) - Static Mixture Model")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "diagnostic_regimes.png"))
    plt.close()

    print(f"\n✅ Walk-forward analysis complete. Results are in '{OUTPUT_DIR}'.")