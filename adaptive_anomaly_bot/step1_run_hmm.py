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

# Suppress warnings for cleaner output
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

# --- 1. Configuration ---
TICKER = "QQQ"
PERIOD = "5y"
OUTPUT_DIR = "hmm_results"
# MCMC sampling parameters (can be increased for more robust results)
DRAWS = 1000
TUNE = 1000
CHAINS = 2

# --- 2. Data Preparation ---
def fetch_data(ticker, period):
    """Fetches historical data and calculates daily returns."""
    print(f"Fetching {period} of data for {ticker}...")
    stock_data = yf.Ticker(ticker).history(period=period, interval="1d")
    returns = stock_data['Close'].pct_change().dropna()
    print(f"Data fetched. Shape of returns: {returns.shape}")
    return returns, stock_data

# --- 3. HMM Model Definition ---
def logp_hmm_forward(p_initial, p_transition, mus, sigmas, obs):
    """Calculates the log-likelihood of an HMM using the forward algorithm."""
    ob_0_probs = pm.logp(pm.Normal.dist(mu=mus[0], sigma=sigmas), obs[0])
    alpha_0 = pm.logp(pm.Categorical.dist(p=p_initial), 0) + ob_0_probs

    def forward_step(obs_t, alpha_tm1, mus_t, sigmas, p_transition):
        ob_t_probs = pm.logp(pm.Normal.dist(mu=mus_t, sigma=sigmas), obs_t)
        alpha_t = pm.logsumexp(alpha_tm1 + pt.log(p_transition.T), axis=1) + ob_t_probs
        return alpha_t

    alphas, _ = scan(
        fn=forward_step,
        sequences=[obs[1:], mus[1:]],
        outputs_info=[alpha_0],
        non_sequences=[sigmas, p_transition]
    )
    return pm.logsumexp(alphas[-1])

# --- 4. Main Execution ---
if __name__ == "__main__":
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Get data
    returns, stock_data = fetch_data(TICKER, PERIOD)
    returns_data = returns.values

    # Build the PyMC Model
    print("Building PyMC HMM Model...")
    with pm.Model() as hmm_model:
        # Priors
        p_transition = pm.Dirichlet("p_transition", a=np.ones((3, 3)), shape=(3, 3))
        sigma = pm.HalfNormal("sigma", sigma=0.05, shape=3)
        delta_bear = pm.HalfNormal("delta_bear", sigma=0.01)
        delta_bull = pm.HalfNormal("delta_bull", sigma=0.01)
        sigma_drift = pm.HalfNormal("sigma_drift", sigma=0.001)
        p_initial = pm.Dirichlet("p_initial", a=np.ones(3))

        # Dynamic baseline and constrained means
        mu_sideways = pm.GaussianRandomWalk("mu_sideways", sigma=sigma_drift, shape=len(returns_data), init_dist=pm.Normal.dist(mu=0, sigma=0.01))
        mu_bear = mu_sideways - delta_bear
        mu_bull = mu_sideways + delta_bull
        mus = pt.stack([mu_bear, mu_sideways, mu_bull], axis=1)

        # Likelihood
        log_likelihood = pm.CustomDist(
            "log_likelihood",
            p_initial,
            p_transition,
            mus,
            sigma,
            observed=returns_data,
            logp=logp_hmm_forward
        )

    print("Model construction complete.")

    # Run the MCMC sampler
    with hmm_model:
        print(f"Starting MCMC sampling ({CHAINS} chains, {DRAWS} draws, {TUNE} tune steps)...")
        idata = pm.sample(draws=DRAWS, tune=TUNE, chains=CHAINS, cores=1, target_accept=0.9)
        print("Sampling complete.")

    # Save results
    print("Saving inference data and model...")
    idata.to_netcdf(os.path.join(OUTPUT_DIR, "hmm_inference_data.nc"))
    with open(os.path.join(OUTPUT_DIR, "hmm_model.pkl"), "wb") as f:
        cloudpickle.dump(hmm_model, f)
    
    # --- 5. Generate Diagnostics ---
    print("Generating diagnostic plots...")

    # Trace Plot
    az.plot_trace(idata, var_names=["p_transition", "sigma", "delta_bear", "delta_bull", "sigma_drift"])
    plt.savefig(os.path.join(OUTPUT_DIR, "diagnostic_trace.png"))
    plt.close()

    # Posterior Plot
    az.plot_posterior(idata, var_names=["sigma", "delta_bear", "delta_bull", "sigma_drift"])
    plt.savefig(os.path.join(OUTPUT_DIR, "diagnostic_posterior.png"))
    plt.close()

    # Regime Visualization Plot
    print("Generating regime visualization plot...")
    # Extract posterior mean values
    posterior_data = idata.posterior.stack(sample=("chain", "draw"))
    mu_sideways_mean = posterior_data["mu_sideways"].mean(axis=-1)
    delta_bear_mean = posterior_data["delta_bear"].mean()
    delta_bull_mean = posterior_data["delta_bull"].mean()
    
    # Create dynamic thresholds
    threshold_bear = mu_sideways_mean - (delta_bear_mean / 2)
    threshold_bull = mu_sideways_mean + (delta_bull_mean / 2)
    
    # Assign regimes based on returns relative to thresholds
    regimes = np.full(len(returns_data), 1) # Default to SIDEWAYS
    regimes[returns_data < threshold_bear] = 0 # BEAR
    regimes[returns_data > threshold_bull] = 2 # BULL
    
    # Plotting
    fig, ax = plt.subplots(figsize=(15, 7))
    plot_data = stock_data.iloc[1:] # Align with returns data
    ax.plot(plot_data.index, plot_data['Close'], color='black', lw=0.5, label=f"{TICKER} Close Price")

    ax.fill_between(plot_data.index, ax.get_ylim()[0], ax.get_ylim()[1], where=regimes == 0, color='red', alpha=0.3, label="BEAR Regime")
    ax.fill_between(plot_data.index, ax.get_ylim()[0], ax.get_ylim()[1], where=regimes == 1, color='gray', alpha=0.3, label="SIDEWAYS Regime")
    ax.fill_between(plot_data.index, ax.get_ylim()[0], ax.get_ylim()[1], where=regimes == 2, color='green', alpha=0.3, label="BULL Regime")

    ax.set_title(f"Market Regimes for {TICKER} identified by Bayesian HMM")
    ax.set_ylabel("Price")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "diagnostic_regimes.png"))
    plt.close()

    print(f"\n✅ All steps complete. Results are saved in the '{OUTPUT_DIR}' folder.")