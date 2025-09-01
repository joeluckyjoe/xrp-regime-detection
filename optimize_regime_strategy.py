import pandas as pd
import numpy as np
import arviz as az
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import warnings

# Import the optimization libraries
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

# (This is the same function from the previous script)
def get_regime_probabilities(trace, features_df):
    """
    Calculates the posterior probability of each data point belonging to each regime
    using a vectorized approach for efficiency.
    """
    print("Calculating daily regime probabilities (this happens only once)...")
    post = trace.posterior
    weights_raw = post["weights"].values
    means_raw = post["means"].values
    chol_corr_raw = post["chol_corr"].values
    sigmas_raw = post["sigmas"].values
    n_samples = weights_raw.shape[0] * weights_raw.shape[1]
    n_regimes = means_raw.shape[2]
    n_features = means_raw.shape[3]
    weights = weights_raw.reshape(n_samples, -1)
    means = means_raw.reshape(n_samples, n_regimes, n_features)
    sigmas = sigmas_raw.reshape(n_samples, n_regimes, n_features)
    chol_corr_flat = chol_corr_raw.reshape(n_samples, -1)
    chol_corr_shared = np.zeros((n_samples, n_features, n_features))
    rows, cols = np.tril_indices(n_features)
    chol_corr_shared[:, rows, cols] = chol_corr_flat
    chol_corr = np.expand_dims(chol_corr_shared, axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_df)
    n_data_points, _ = X_scaled.shape
    from scipy.linalg import solve_triangular
    from scipy.special import logsumexp
    sigmas_reshaped = sigmas[..., np.newaxis]
    chol_covs = sigmas_reshaped * chol_corr
    regime_probs = np.zeros((n_data_points, n_regimes))
    for i in range(n_data_points):
        x_i = X_scaled[i]
        diagonals = np.einsum('srii->sr', chol_covs)
        log_det_cov = 2 * np.sum(np.log(diagonals), axis=1)
        diff = x_i - means
        y = solve_triangular(chol_covs, diff[..., np.newaxis], lower=True).squeeze(axis=-1)
        mahalanobis_dist = np.sum(y**2, axis=2)
        log_likelihoods_i = -0.5 * (n_features * np.log(2 * np.pi) + log_det_cov[:, np.newaxis] + mahalanobis_dist)
        log_probs = np.log(weights) + log_likelihoods_i
        log_probs_normalized = log_probs - logsumexp(log_probs, axis=1, keepdims=True)
        probs = np.exp(log_probs_normalized)
        regime_probs[i, :] = probs.mean(axis=0)
    return pd.DataFrame(regime_probs, index=features_df.index)


def run_backtest(regime_probs, price_data, buy_threshold, sell_threshold, exit_threshold, transaction_cost=0.001):
    """
    Runs a trading simulation with a "hold-the-signal" logic.
    """
    aligned_prices = price_data.reindex(regime_probs.index).dropna()
    regime_probs = regime_probs.reindex(aligned_prices.index)
    position = 0
    pnl = 0
    trade_log = []
    
    for i in range(1, len(aligned_prices)):
        prob_buy = regime_probs.iloc[i-1, 0]
        prob_sell = regime_probs.iloc[i-1, 2]
        current_price = aligned_prices['Open'].iloc[i]

        # --- NEW TRADING LOGIC ---
        # 1. Check if we should EXIT a LONG position
        if position == 1 and prob_buy < exit_threshold:
            trade_pnl = (current_price - trade_log[-1]['price']) * position - (abs(current_price) + abs(trade_log[-1]['price'])) * transaction_cost
            pnl += trade_pnl
            trade_log[-1]['pnl'] = trade_pnl
            position = 0
        
        # 2. Check if we should EXIT a SHORT position
        elif position == -1 and prob_sell < exit_threshold:
            trade_pnl = (current_price - trade_log[-1]['price']) * position - (abs(current_price) + abs(trade_log[-1]['price'])) * transaction_cost
            pnl += trade_pnl
            trade_log[-1]['pnl'] = trade_pnl
            position = 0

        # 3. Check if we should ENTER a new position (only if we are flat)
        if position == 0:
            if prob_buy > buy_threshold:
                position = 1
                trade_log.append({'date': aligned_prices.index[i], 'type': 'BUY', 'price': current_price, 'pnl': np.nan})
            elif prob_sell > sell_threshold:
                position = -1
                trade_log.append({'date': aligned_prices.index[i], 'type': 'SELL', 'price': current_price, 'pnl': np.nan})
        # --- END NEW LOGIC ---

    return pnl, pd.DataFrame(trade_log)

# --- BAYESIAN OPTIMIZATION SETUP ---

# 1. Define the parameter space to search (added exit_threshold)
param_space = [
    Real(0.50, 0.99, name='buy_threshold'),
    Real(0.50, 0.99, name='sell_threshold'),
    Real(0.40, 0.60, name='exit_threshold') # Exit if conviction drops below this level
]

# 2. Load and prepare all necessary data ONCE
print("Loading model, features, and price data...")
trace = az.from_netcdf('data/regime_model_trace.nc')
features = pd.read_csv('data/informed_flow_features.csv', index_col=0, parse_dates=True)

# Calculate regime probabilities once to use in all optimization calls
regime_probabilities = get_regime_probabilities(trace, features)
regime_probabilities.columns = ['Prob_Buy', 'Prob_Normal', 'Prob_Sell']

# Fetch price data once to use in all optimization calls
end_date = regime_probabilities.index.max() + timedelta(days=1)
start_date = regime_probabilities.index.min()
price_data = yf.download("QQQ", start=start_date, end=end_date, interval='1d')
if isinstance(price_data.columns, pd.MultiIndex):
    price_data.columns = price_data.columns.get_level_values(0)

# 3. Define the objective function for the optimizer
@use_named_args(param_space)
def objective(buy_threshold, sell_threshold, exit_threshold):
    """
    This function takes the parameters, runs the new backtest, and returns the score.
    """
    pnl, _ = run_backtest(
        regime_probs=regime_probabilities,
        price_data=price_data,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        exit_threshold=exit_threshold
    )
    return -pnl

# --- RUN THE OPTIMIZATION ---
if __name__ == '__main__':
    print("\nStarting Bayesian Optimization with new 'hold-the-signal' logic...")
    print("This will run multiple backtests. Please wait...")
    
    result = gp_minimize(objective, param_space, n_calls=75, random_state=0, n_jobs=2) # Increased n_calls for 3 params
    
    best_pnl = -result.fun
    best_buy_threshold, best_sell_threshold, best_exit_threshold = result.x

    print("\n" + "="*50)
    print("      SMART MONEY STRATEGY OPTIMIZATION COMPLETE")
    print("="*50)
    print(f"Best PNL Found:      ${best_pnl:.2f}")
    print("\nOptimal Parameters:")
    print(f"- Best Buy Threshold:  {best_buy_threshold:.4f}")
    print(f"- Best Sell Threshold: {best_sell_threshold:.4f}")
    print(f"- Best Exit Threshold: {best_exit_threshold:.4f} (Exit when signal drops below this)")
    print("="*50)
