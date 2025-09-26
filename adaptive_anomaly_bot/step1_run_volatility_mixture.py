import os
import warnings
import yfinance as yf
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
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
VOLATILITY_WINDOW = 24 # Use 24-hour rolling volatility
OUTPUT_DIR = "volatility_model_results"
DRAWS = 2000
TUNE = 2000
CHAINS = 4

# --- 2. Data Preparation ---
def fetch_and_prepare_data(ticker, period, interval, vol_window):
    print(f"Fetching {period} of {interval} data for {ticker}...")
    stock_data = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
    returns = stock_data['Close'].pct_change()
    
    # Calculate rolling volatility as our new input feature
    volatility = returns.rolling(window=vol_window).std().dropna()
    
    # Align stock_data with the new volatility series
    aligned_stock_data = stock_data.loc[volatility.index]

    print(f"Data prepared. Shape of volatility series: {volatility.shape}")
    return volatility, aligned_stock_data

# --- 3. Prediction Function ---
def get_regime_probabilities(vol_data, model_params):
    mus = model_params['mus']
    sigmas = model_params['sigmas']
    p_regime = model_params['p_regime']
    
    vol_col = vol_data[:, np.newaxis]
    like_matrix = pm.logp(pm.Normal.dist(mu=mus, sigma=sigmas), vol_col).eval()
    log_probs = np.log(p_regime) + like_matrix
    return np.argmax(log_probs, axis=1)

# --- 4. Main Execution ---
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    volatility, stock_data = fetch_and_prepare_data(TICKER, PERIOD, INTERVAL, VOLATILITY_WINDOW)
    
    split_idx = int(len(volatility) * TRAIN_SPLIT_RATIO)
    train_vol = volatility.values[:split_idx]
    test_vol = volatility.values[split_idx:]
    print(f"Data split: Training set size={len(train_vol)}, Test set size={len(test_vol)}")

    with pm.Model() as volatility_model:
        p_regime = pm.Dirichlet("p_regime", a=np.ones(3))
        
        # Priors for the standard deviation of volatility in each regime
        sigma = pm.HalfNormal("sigma", sigma=0.01, shape=3)

        # Priors for the MEAN VOLATILITY of each regime
        # Using a HalfNormal as volatility must be positive
        mus = pm.HalfNormal(
            "mus", 
            sigma=0.01, 
            shape=3, 
            transform=pm.distributions.transforms.ordered,
            initval=np.array([0.001, 0.005, 0.01]) # Sensible starting vols
        )
        
        comp_dists = pm.Normal.dist(mu=mus, sigma=sigma)
        
        likelihood = pm.Mixture(
            "likelihood",
            w=p_regime,
            comp_dists=comp_dists,
            observed=train_vol[:, None]
        )

    with volatility_model:
        print("Starting MCMC sampling on training data...")
        idata = pm.sample(draws=DRAWS, tune=TUNE, chains=CHAINS, cores=1, target_accept=0.9)
    
    print("Saving inference data and model...")
    idata.to_netcdf(os.path.join(OUTPUT_DIR, "volatility_inference_data.nc"))
    with open(os.path.join(OUTPUT_DIR, "volatility_model.pkl"), "wb") as f: cloudpickle.dump(volatility_model, f)
    
    posterior_data = idata.posterior.stack(sample=("chain", "draw"))
    model_params = {
        'p_regime': posterior_data['p_regime'].mean(axis=-1).values,
        'sigmas': posterior_data['sigma'].mean(axis=-1).values,
        'mus': posterior_data['mus'].mean(axis=-1).values
    }

    print("Predicting regimes on train and test data...")
    train_regimes = get_regime_probabilities(train_vol, model_params)
    test_regimes = get_regime_probabilities(test_vol, model_params)
    all_regimes = np.concatenate([train_regimes, test_regimes])
    
    print("Generating diagnostic plots...")
    az.plot_trace(idata, var_names=["p_regime", "sigma", "mus"])
    plt.savefig(os.path.join(OUTPUT_DIR, "diagnostic_trace.png"))
    plt.close()
    
    fig, ax = plt.subplots(figsize=(15, 7))
    plot_data = stock_data
    ax.plot(plot_data.index, plot_data['Close'], color='black', lw=0.5)
    
    # Regimes are now LOW_VOL, MID_VOL, HIGH_VOL
    ax.fill_between(plot_data.index, ax.get_ylim()[0], ax.get_ylim()[1], where=all_regimes==0, color='green', alpha=0.3, label="LOW_VOL Regime")
    ax.fill_between(plot_data.index, ax.get_ylim()[0], ax.get_ylim()[1], where=all_regimes==1, color='gray', alpha=0.3, label="MID_VOL Regime")
    ax.fill_between(plot_data.index, ax.get_ylim()[0], ax.get_ylim()[1], where=all_regimes==2, color='red', alpha=0.3, label="HIGH_VOL Regime")
    
    split_time = plot_data.index[split_idx]
    ax.axvline(x=split_time, color='blue', linestyle='--', lw=2, label='Train/Test Split')
    ax.set_title(f"Volatility Regimes for {TICKER} (Hourly) - Static Mixture Model")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "diagnostic_regimes.png"))
    plt.close()

    print(f"\n✅ Walk-forward analysis complete. Results are in '{OUTPUT_DIR}'.")