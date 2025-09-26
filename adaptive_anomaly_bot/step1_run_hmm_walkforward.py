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
PERIOD = "729d" # Max allowed for 1h interval
INTERVAL = "1h"
TRAIN_SPLIT_RATIO = 0.7
OUTPUT_DIR = "mixture_model_results_v2" # New folder for new results
DRAWS = 1500
TUNE = 1500
CHAINS = 2

# --- 2. Data Preparation ---
def fetch_data(ticker, period, interval):
    print(f"Fetching {period} of {interval} data for {ticker}...")
    stock_data = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
    returns = stock_data['Close'].pct_change().dropna()
    print(f"Data fetched. Shape of returns: {returns.shape}")
    return returns, stock_data

# --- 3. Prediction Function ---
def get_regime_probabilities(returns, model_params):
    """Calculates the probability of each data point belonging to each regime."""
    n_obs = len(returns)
    rolling_mean_baseline = pd.Series(returns).rolling(window=21*7, min_periods=1).mean().values
    
    bear_means = rolling_mean_baseline - model_params['delta_bear']
    bull_means = rolling_mean_baseline + model_params['delta_bull']
    
    like_bear = pm.logp(pm.Normal.dist(mu=bear_means, sigma=model_params['sigma'][0]), returns).eval()
    like_sideways = pm.logp(pm.Normal.dist(mu=rolling_mean_baseline, sigma=model_params['sigma'][1]), returns).eval()
    like_bull = pm.logp(pm.Normal.dist(mu=bull_means, sigma=model_params['sigma'][2]), returns).eval()
    
    probs_bear = np.log(model_params['p_regime'][0]) + like_bear
    probs_sideways = np.log(model_params['p_regime'][1]) + like_sideways
    probs_bull = np.log(model_params['p_regime'][2]) + like_bull
    
    all_probs = np.stack([probs_bear, probs_sideways, probs_bull])
    return np.argmax(all_probs, axis=0)

# --- 4. Main Execution ---
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    returns, stock_data = fetch_data(TICKER, PERIOD, INTERVAL)
    split_idx = int(len(returns) * TRAIN_SPLIT_RATIO)
    train_returns = returns.values[:split_idx]
    test_returns = returns.values[split_idx:]
    print(f"Data split: Training set size={len(train_returns)}, Test set size={len(test_returns)}")

    with pm.Model() as mixture_model:
        p_regime = pm.Dirichlet("p_regime", a=np.ones(3))
        sigma = pm.HalfNormal("sigma", sigma=0.05, shape=3)
        sigma_drift = pm.HalfNormal("sigma_drift", sigma=0.001)

        # --- FIX: Use more informative LogNormal priors for the delta offsets ---
        # The parameters here (mu=-5, sigma=1) encourage values around exp(-5) ~ 0.0067
        # This keeps the offsets small but discourages them from collapsing to zero.
        delta_bear = pm.LogNormal("delta_bear", mu=-5, sigma=1.0)
        delta_bull = pm.LogNormal("delta_bull", mu=-5, sigma=1.0)
        
        mu_sideways = pm.GaussianRandomWalk("mu_sideways", sigma=sigma_drift, shape=len(train_returns), init_dist=pm.Normal.dist(mu=0, sigma=0.01))
        
        comp_dists = pm.Normal.dist(
            mu=pt.stack([
                mu_sideways - delta_bear,
                mu_sideways,
                mu_sideways + delta_bull
            ]).T,
            sigma=sigma,
            shape=(len(train_returns), 3)
        )
        
        likelihood = pm.Mixture(
            "likelihood",
            w=p_regime,
            comp_dists=comp_dists,
            observed=train_returns
        )

    with mixture_model:
        print("Starting MCMC sampling on training data...")
        idata = pm.sample(draws=DRAWS, tune=TUNE, chains=CHAINS, cores=1, target_accept=0.95)
    
    print("Saving inference data and model...")
    idata.to_netcdf(os.path.join(OUTPUT_DIR, "mixture_inference_data.nc"))
    with open(os.path.join(OUTPUT_DIR, "mixture_model.pkl"), "wb") as f: cloudpickle.dump(mixture_model, f)
    
    posterior_data = idata.posterior.stack(sample=("chain", "draw"))
    model_params = {
        'p_regime': posterior_data['p_regime'].mean(axis=-1).values,
        'sigma': posterior_data['sigma'].mean(axis=-1).values,
        'delta_bear': posterior_data['delta_bear'].mean().item(),
        'delta_bull': posterior_data['delta_bull'].mean().item()
    }

    print("Predicting regimes on train and test data...")
    train_regimes = get_regime_probabilities(train_returns, model_params)
    test_regimes = get_regime_probabilities(test_returns, model_params)
    all_regimes = np.concatenate([train_regimes, test_regimes])
    
    print("Generating diagnostic plots...")
    az.plot_trace(idata, var_names=["p_regime", "sigma", "delta_bear", "delta_bull"])
    plt.savefig(os.path.join(OUTPUT_DIR, "diagnostic_trace.png"))
    plt.close()
    
    fig, ax = plt.subplots(figsize=(15, 7))
    plot_data = stock_data.iloc[1:]
    ax.plot(plot_data.index, plot_data['Close'], color='black', lw=0.5)
    ax.fill_between(plot_data.index, ax.get_ylim()[0], ax.get_ylim()[1], where=all_regimes==0, color='red', alpha=0.3, label="BEAR")
    ax.fill_between(plot_data.index, ax.get_ylim()[0], ax.get_ylim()[1], where=all_regimes==1, color='gray', alpha=0.3, label="SIDEWAYS")
    ax.fill_between(plot_data.index, ax.get_ylim()[0], ax.get_ylim()[1], where=all_regimes==2, color='green', alpha=0.3, label="BULL")
    split_time = plot_data.index[split_idx]
    ax.axvline(x=split_time, color='blue', linestyle='--', lw=2, label='Train/Test Split')
    ax.set_title(f"Walk-Forward Market Regimes for {TICKER} (Hourly) - Dynamic Mixture Model v2")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "diagnostic_regimes.png"))
    plt.close()

    print(f"\n✅ Walk-forward analysis complete. Results are in '{OUTPUT_DIR}'.")