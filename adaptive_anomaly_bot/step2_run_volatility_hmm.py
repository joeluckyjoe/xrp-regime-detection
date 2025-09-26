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
from pytensor.scan.basic import scan

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
OUTPUT_DIR = "hmm_volatility_results"
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

# --- 3. HMM Likelihood and Viterbi Decoder ---
def logp_hmm_forward(p_initial, p_transition, mus, sigmas, obs):
    p_initial_flat = p_initial.flatten()
    sigmas_flat = sigmas.flatten()
    
    def forward_step(t, obs_t, alpha_tm1, mus, sigmas, p_transition):
        mus_t = mus[t] # This indexing is symbolic
        ob_t_probs = pm.logp(pm.Normal.dist(mu=mus_t, sigma=sigmas), obs_t)
        alpha_t = pm.logsumexp(alpha_tm1 + pt.log(p_transition.T), axis=1) + ob_t_probs
        return alpha_t

    ob_0_probs = pm.logp(pm.Normal.dist(mu=mus[0], sigma=sigmas_flat), obs[0])
    alpha_0 = pm.logp(pm.Categorical.dist(p=p_initial_flat), 0) + ob_0_probs
    
    indices = pt.arange(1, obs.shape[0])
    
    alphas, _ = scan(
        fn=forward_step,
        sequences=[indices, obs[1:]],
        outputs_info=[alpha_0],
        non_sequences=[mus, sigmas_flat, p_transition]
    )
    return pm.logsumexp(alphas[-1])

def viterbi_decode(X, model_params):
    n_obs = len(X)
    n_states = model_params['p_initial'].shape[0]
    viterbi_trellis = np.zeros((n_obs, n_states))
    backpointers = np.zeros((n_obs, n_states), dtype=int)
    
    mus = model_params['mus']
    sigmas = model_params['sigmas']
    p_initial = model_params['p_initial']
    p_transition = model_params['p_transition']
    
    emission_probs = np.array([pm.logp(pm.Normal.dist(mu=m, sigma=s), X[0]).eval() for m, s in zip(mus, sigmas)]).flatten()
    viterbi_trellis[0, :] = np.log(p_initial) + emission_probs

    for t in range(1, n_obs):
        emission_probs = np.array([pm.logp(pm.Normal.dist(mu=m, sigma=s), X[t]).eval() for m, s in zip(mus, sigmas)]).flatten()
        for k in range(n_states):
            path_probs = viterbi_trellis[t-1, :] + np.log(p_transition[:, k])
            backpointers[t, k] = np.argmax(path_probs)
            viterbi_trellis[t, k] = np.max(path_probs) + emission_probs[k]

    best_path = np.zeros(n_obs, dtype=int)
    best_path[-1] = np.argmax(viterbi_trellis[-1, :])
    for t in range(n_obs - 2, -1, -1):
        best_path[t] = backpointers[t + 1, best_path[t + 1]]
    return best_path

# --- 4. Main Execution ---
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    volatility, stock_data = fetch_and_prepare_data(TICKER, PERIOD, INTERVAL, VOLATILITY_WINDOW)
    
    split_idx = int(len(volatility) * TRAIN_SPLIT_RATIO)
    train_vol = volatility.values[:split_idx]
    test_vol = volatility.values[split_idx:]
    print(f"Data split: Training set size={len(train_vol)}, Test set size={len(test_vol)}")

    with pm.Model() as hmm_vol_model:
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
        
        # The HMM likelihood, added directly to the model's logp
        # NOTE: For a static HMM, `mus` is not time-varying, so we tile it.
        mus_tiled = pt.tile(mus, (len(train_vol), 1))
        hmm_loglike = logp_hmm_forward(p_initial, p_transition, mus_tiled, sigma, train_vol)
        pm.Potential("hmm_likelihood", hmm_loglike)

    with hmm_vol_model:
        print("Starting MCMC sampling on training data...")
        idata = pm.sample(draws=DRAWS, tune=TUNE, chains=CHAINS, cores=1, target_accept=0.9)
    
    print("Saving inference data and model...")
    idata.to_netcdf(os.path.join(OUTPUT_DIR, "hmm_volatility_inference_data.nc"))
    with open(os.path.join(OUTPUT_DIR, "hmm_volatility_model.pkl"), "wb") as f: cloudpickle.dump(hmm_vol_model, f)
    
    posterior_data = idata.posterior.stack(sample=("chain", "draw"))
    model_params = {
        'p_initial': posterior_data['p_initial'].mean(axis=-1).values,
        'p_transition': posterior_data['p_transition'].mean(axis=-1).values,
        'sigmas': posterior_data['sigma'].mean(axis=-1).values,
        'mus': posterior_data['mus'].mean(axis=-1).values
    }

    print("Predicting regimes on train and test data...")
    train_regimes = viterbi_decode(train_vol, model_params)
    test_regimes = viterbi_decode(test_vol, model_params)
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
    ax.set_title(f"Volatility HMM Regimes for {TICKER} (Hourly)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "diagnostic_regimes.png"))
    plt.close()

    print(f"\n✅ Walk-forward analysis complete. Results are in '{OUTPUT_DIR}'.")