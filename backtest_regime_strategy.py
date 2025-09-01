import pandas as pd
import numpy as np
import arviz as az
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

def get_regime_probabilities(trace, features_df):
    """
    Calculates the posterior probability of each data point belonging to each regime
    using a vectorized approach for efficiency.
    """
    print("Calculating daily regime probabilities...")

    # Extract posterior samples from the trace
    post = trace.posterior

    # --- FIX: Reshape posterior samples robustly ---
    # Get the raw values from the trace
    weights_raw = post["weights"].values
    means_raw = post["means"].values
    chol_corr_raw = post["chol_corr"].values
    sigmas_raw = post["sigmas"].values

    # Reshape to combine chain and draw dimensions into a single 'sample' dimension at the front
    n_samples = weights_raw.shape[0] * weights_raw.shape[1]
    n_regimes = means_raw.shape[2]
    n_features = means_raw.shape[3]

    weights = weights_raw.reshape(n_samples, -1)
    means = means_raw.reshape(n_samples, n_regimes, n_features)
    sigmas = sigmas_raw.reshape(n_samples, n_regimes, n_features)
    
    # The LKJ distribution returns a flattened array of the lower triangular elements.
    # The trace indicates a SHARED correlation matrix was saved. We reconstruct it.
    chol_corr_flat = chol_corr_raw.reshape(n_samples, -1)
    chol_corr_shared = np.zeros((n_samples, n_features, n_features))
    rows, cols = np.tril_indices(n_features)
    chol_corr_shared[:, rows, cols] = chol_corr_flat
    
    # Add a new axis for the regimes so it can be broadcast during multiplication
    chol_corr = np.expand_dims(chol_corr_shared, axis=1)
    # --- END FIX ---

    # Scale the features exactly as they were for training
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_df)
    n_data_points, _ = X_scaled.shape

    # Import necessary libraries for calculation
    from scipy.linalg import solve_triangular
    from scipy.special import logsumexp

    # Reconstruct the Cholesky factor of the full covariance matrix
    # This is vectorized across all posterior samples and regimes
    sigmas_reshaped = sigmas[..., np.newaxis]
    chol_covs = sigmas_reshaped * chol_corr

    # This array will store the final probability of each regime for each day
    regime_probs = np.zeros((n_data_points, n_regimes))

    # Loop through each day of data
    for i in range(n_data_points):
        x_i = X_scaled[i]
        
        # --- Start of vectorized log-probability calculation ---
        # Calculate the log determinant of the covariance matrices
        # The einsum pattern 'srii->sr' extracts the diagonals for each sample and regime
        diagonals = np.einsum('srii->sr', chol_covs)
        log_det_cov = 2 * np.sum(np.log(diagonals), axis=1)

        # Solve for L_inv * (x - mu) using the Cholesky factor
        diff = x_i - means
        y = solve_triangular(chol_covs, diff[..., np.newaxis], lower=True).squeeze(axis=-1)
        
        # Calculate the Mahalanobis distance
        mahalanobis_dist = np.sum(y**2, axis=2)

        # Combine terms to get the log-likelihood for each regime and posterior sample
        log_likelihoods_i = -0.5 * (n_features * np.log(2 * np.pi) + log_det_cov[:, np.newaxis] + mahalanobis_dist)
        # --- End of vectorized calculation ---

        # Combine with regime weights in log space
        log_probs = np.log(weights) + log_likelihoods_i
        
        # Normalize to get probabilities (using log-sum-exp for numerical stability)
        log_probs_normalized = log_probs - logsumexp(log_probs, axis=1, keepdims=True)
        probs = np.exp(log_probs_normalized)
        
        # Average across all posterior samples to get the final probability for the day
        regime_probs[i, :] = probs.mean(axis=0)

    return pd.DataFrame(regime_probs, index=features_df.index)

def run_backtest(regime_probs, ticker="QQQ", buy_threshold=0.7, sell_threshold=0.7, transaction_cost=0.001):
    """
    Runs a trading simulation based on the regime probabilities.
    """
    print("Running backtest simulation...")
    
    # Fetch price data for the backtest period
    end_date = regime_probs.index.max() + timedelta(days=1) # Add one day to ensure we get the last day's data
    start_date = regime_probs.index.min()
    price_data = yf.download(ticker, start=start_date, end=end_date, interval='1d')
    if isinstance(price_data.columns, pd.MultiIndex):
        price_data.columns = price_data.columns.get_level_values(0)

    # Align price data with our probabilities
    aligned_prices = price_data.reindex(regime_probs.index).dropna()
    regime_probs = regime_probs.reindex(aligned_prices.index)

    position = 0
    pnl = 0
    trade_log = []

    for i in range(1, len(aligned_prices)):
        # Get probabilities for the previous day to make a decision for today
        prob_buy = regime_probs.iloc[i-1, 0] # Assumes Regime 0 is "Informed Buying"
        prob_sell = regime_probs.iloc[i-1, 2] # Assumes Regime 2 is "Informed Selling"
        
        entry_price = aligned_prices['Open'].iloc[i]
        
        # Close any open position at today's open
        if position != 0:
            trade_pnl = (entry_price - trade_log[-1]['price']) * position - (abs(entry_price) + abs(trade_log[-1]['price'])) * transaction_cost
            pnl += trade_pnl
            trade_log[-1]['pnl'] = trade_pnl
            position = 0

        # Decide whether to open a new position
        if prob_buy > buy_threshold:
            position = 1 # Go Long
            trade_log.append({'date': aligned_prices.index[i], 'type': 'BUY', 'price': entry_price, 'pnl': np.nan})
        elif prob_sell > sell_threshold:
            position = -1 # Go Short
            trade_log.append({'date': aligned_prices.index[i], 'type': 'SELL', 'price': entry_price, 'pnl': np.nan})

    return pnl, pd.DataFrame(trade_log)


if __name__ == '__main__':
    try:
        trace = az.from_netcdf('data/regime_model_trace.nc')
        features = pd.read_csv('data/informed_flow_features.csv', index_col=0, parse_dates=True)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        print("Please ensure you have run the previous two scripts successfully.")
    else:
        # Step 1: Calculate the daily regime probabilities from the trained model
        regime_probabilities = get_regime_probabilities(trace, features)
        
        # --- IMPORTANT: Identify the regimes ---
        # We need to map the regime index (0, 1, 2) to our concepts
        # (Buy, Sell, Normal) by looking at the model summary from the last script.
        # Based on your previous output:
        # Regime 0 = Informed Buying (High volume spike, positive price-vol interaction)
        # Regime 1 = Normal
        # Regime 2 = Informed Selling (High volume spike, negative price-vol interaction)
        regime_probabilities.columns = ['Prob_Buy', 'Prob_Normal', 'Prob_Sell']
        print("\nRegime Probabilities Calculated. Preview:")
        print(regime_probabilities.tail())

        # Step 2: Run the backtest using these probabilities
        total_pnl, trades = run_backtest(regime_probabilities)

        # Step 3: Print the results
        print("\n" + "="*50)
        print("          SMART MONEY STRATEGY BACKTEST RESULTS")
        print("="*50)

        if not trades.empty:
            closed_trades = trades.dropna(subset=['pnl'])
            winning_trades = closed_trades[closed_trades['pnl'] > 0]
            
            print(f"Total Closed Trades: {len(closed_trades)}")
            if len(closed_trades) > 0:
                print(f"Winning Trades:      {len(winning_trades)}")
                win_rate = len(winning_trades) / len(closed_trades)
                print(f"Win Rate:            {win_rate:.2%}")
        else:
            print("No trades were executed.")

        print(f"\nTotal PNL (USD):     ${total_pnl:.2f} (per 1 unit of QQQ)")
        print("="*50)
        
        if not trades.dropna().empty:
            print("\nLast 10 Trades:")
            print(trades.dropna().tail(10))

