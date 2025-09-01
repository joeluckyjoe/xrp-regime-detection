import pandas as pd
import pymc as pm
import numpy as np
import arviz as az
import warnings
from sklearn.preprocessing import StandardScaler
import yfinance as yf
from datetime import datetime, timedelta
import traceback

# Import the necessary optimization libraries
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

warnings.filterwarnings("ignore")

def fetch_yfinance_data(ticker, start_date, end_date):
    """
    Fetches and cleans the historical OHLC data, handling yfinance's
    potential MultiIndex column format.
    """
    df = yf.download(ticker, start=start_date, end=end_date, interval='1d')
    if df.empty: return df

    # Flatten the column index if yfinance returns a MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Now we can safely rename the columns
    df.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    }, inplace=True)
    
    df.columns = df.columns.str.lower()

    if df.index.tz is None:
        df = df.tz_localize('UTC')
    return df

def run_bayesian_backtest(trace, features_df, price_data, buy_threshold, sell_threshold, transaction_cost=0.001):
    """
    Runs a backtest using the trained Bayesian model. This is our core evaluation function.
    It takes thresholds as arguments.
    """
    # Note: Data is assumed to be pre-aligned and clean before entering this function
    scaler = StandardScaler()
    X = scaler.fit_transform(features_df)

    betas = az.extract(trace, var_names=["betas"]).values
    intercept = az.extract(trace, var_names=["intercept"]).values

    all_logits = np.dot(X, betas) + intercept
    all_probabilities = 1 / (1 + np.exp(-all_logits))
    mean_probabilities = all_probabilities.mean(axis=1)

    position = 0
    pnl = 0
    trade_log = []

    for i in range(len(mean_probabilities) - 1):
        prob_up = mean_probabilities[i]

        if position != 0:
            exit_price_row = price_data.iloc[i + 1]
            exit_price = exit_price_row['open']
            if pd.isna(exit_price):
                position = 0
                continue
            entry_price = trade_log[-1]['price']
            trade_pnl = (exit_price - entry_price) * position - (abs(exit_price) + abs(entry_price)) * transaction_cost
            pnl += trade_pnl
            for log in reversed(trade_log):
                if pd.isna(log['pnl']):
                    log['pnl'] = trade_pnl
                    break
            position = 0

        entry_price_row = price_data.iloc[i + 1]
        entry_price = entry_price_row['open']
        if pd.isna(entry_price):
            continue

        if prob_up > buy_threshold:
            position = 1
            trade_log.append({'date': entry_price_row.name, 'type': 'BUY', 'price': entry_price, 'pnl': np.nan})
        elif (1 - prob_up) > sell_threshold:
            position = -1
            trade_log.append({'date': entry_price_row.name, 'type': 'SELL', 'price': entry_price, 'pnl': np.nan})

    return pnl, pd.DataFrame(trade_log)

# --- BAYESIAN OPTIMIZATION SETUP ---

# 1. Define the parameter space to search
# We search for the best probability threshold between 50% (random) and 95% (high confidence)
param_space = [
    Real(0.50, 0.95, name='buy_threshold'),
    Real(0.50, 0.95, name='sell_threshold')
]

# 2. Load and prepare data ONCE to be used in all optimization calls
print("Loading and preparing data for optimization...")
trace = az.from_netcdf("data/bayesian_signal_model.nc")
features_unaligned = pd.read_csv('data/nasdaq_features.csv', index_col=0, parse_dates=True)
if features_unaligned.index.tz is None:
    features_unaligned = features_unaligned.tz_localize('UTC')

end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)
start_str, end_str = start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
price_data_unaligned = fetch_yfinance_data("QQQ", start_str, end_str)

# Perform the robust data alignment and cleaning
price_data = price_data_unaligned[~price_data_unaligned.index.duplicated(keep='first')]
features = features_unaligned[~features_unaligned.index.duplicated(keep='first')]
common_index = price_data.index.intersection(features.index)
price_data = price_data.loc[common_index]
features = features.loc[common_index]
print(f"Data aligned. Shape of features: {features.shape}, Shape of price_data: {price_data.shape}")

# 3. Define the objective function for the optimizer
@use_named_args(param_space)
def objective(buy_threshold, sell_threshold):
    """
    This function takes the parameters, runs the backtest, and returns the score to minimize.
    """
    # The backtest function returns (pnl, trades_df), we only need the pnl
    pnl, _ = run_bayesian_backtest(
        trace=trace,
        features_df=features,
        price_data=price_data,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold
    )
    
    # gp_minimize tries to find the minimum value, so we return negative PNL to maximize PNL
    return -pnl

# --- RUN THE OPTIMIZATION ---
if __name__ == '__main__':
    print("\nStarting Bayesian Optimization...")
    print("This will run multiple backtests to find the best thresholds. Please wait...")
    
    # n_calls is the number of backtests the optimizer will run. 50 is a good starting point.
    # n_jobs=-1 uses all available CPU cores to speed up the process.
    result = gp_minimize(objective, param_space, n_calls=50, random_state=0, n_jobs=2)
    
    best_pnl = -result.fun
    best_buy_threshold = result.x[0]
    best_sell_threshold = result.x[1]

    print("\n" + "="*50)
    print("          BAYESIAN STRATEGY OPTIMIZATION COMPLETE")
    print("="*50)
    print(f"Best PNL Found:      ${best_pnl:.2f}")
    print("\nOptimal Parameters:")
    print(f"- Best Buy Threshold:  {best_buy_threshold:.4f} (Enter a BUY when probability > {best_buy_threshold:.1%})")
    print(f"- Best Sell Threshold: {best_sell_threshold:.4f} (Enter a SELL when probability > {best_sell_threshold:.1%})")
    print("="*50)