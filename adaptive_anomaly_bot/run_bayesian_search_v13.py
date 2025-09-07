import pandas as pd
import numpy as np
import os
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

# --- Import the core logic from your existing files ---
# (Ensure these files are in the same directory)
from train_rl_agent import fetch_and_prepare_data
from run_walk_forward_backtest import calculate_walk_forward_features, run_walk_forward_analysis

# --- 1. Define the Search Space ---
# Based on our plan to search around the v12 champion's parameters
SEARCH_SPACE = [
    Real(low=1.25, high=1.75, name='stop_loss_atr_multiplier'),
    Real(low=0.030, high=0.055, name='trend_reward_bonus')
]

LOG_FILE = 'optimization_log_v13.csv'

# --- 2. Prepare the Global Data ---
# We fetch and prepare data once to avoid re-downloading in every optimization step.
print("Loading and preparing all historical data for the optimization run...")
full_data_df = fetch_and_prepare_data(ticker="QQQ", period="729d", interval="1h")
if full_data_df is not None:
    feature_window_size = 441
    all_features = calculate_walk_forward_features(full_data_df)
    aligned_df = full_data_df.iloc[feature_window_size:]
    print("\nData preparation complete. Ready to start optimization.")
else:
    print("\nCRITICAL: Could not fetch data. Exiting.")
    exit()

# --- 3. Define the Objective Function ---
# This function takes parameters and returns a score to be MINIMIZED.
# We return the *negative* portfolio value because skopt minimizes by default.

@use_named_args(SEARCH_SPACE)
def objective_function(stop_loss_atr_multiplier, trend_reward_bonus):
    """
    Runs a full walk-forward backtest for a given set of parameters
    and returns the negative final portfolio value.
    """
    print("\n----------------------------------------------------")
    print(f"Testing Parameters:")
    print(f"  - stop_loss_atr_multiplier: {stop_loss_atr_multiplier:.4f}")
    print(f"  - trend_reward_bonus: {trend_reward_bonus:.4f}")
    
    # Use the v12 trade_penalty and update the two being tuned
    params = {
        'trade_penalty': 0.0954,
        'stop_loss_atr_multiplier': stop_loss_atr_multiplier,
        'trend_reward_bonus': trend_reward_bonus
    }

    # The aligned_df and all_features are accessible from the global scope
    results_df = run_walk_forward_analysis(aligned_df, all_features, num_episodes=150, params=params)

    if results_df.empty:
        final_value = 0
    else:
        final_value = results_df['PortfolioValue'].iloc[-1]
    
    print(f"\n===> Result: Final Portfolio Value = ${final_value:,.2f}")
    print("----------------------------------------------------")

    # Append result to the log file immediately
    with open(LOG_FILE, 'a') as f:
        f.write(f"{stop_loss_atr_multiplier},{trend_reward_bonus},{final_value}\n")

    # Return negative value for minimization
    return -final_value

# --- 4. Main Execution Logic ---
if __name__ == '__main__':
    x0 = []  # Previously tested parameter values
    y0 = []  # Results of previously tested parameters

    print("\n--- Bayesian Optimization for v13 Champion ---")
    if not os.path.exists(LOG_FILE):
        print("Log file not found. Starting a new optimization.")
        # Create the log file with a header
        with open(LOG_FILE, 'w') as f:
            f.write("stop_loss_atr_multiplier,trend_reward_bonus,final_portfolio_value\n")
    else:
        print(f"Log file '{LOG_FILE}' found. Attempting to resume optimization.")
        log_df = pd.read_csv(LOG_FILE)
        if not log_df.empty:
            # Load data for warm-start
            x0 = log_df[['stop_loss_atr_multiplier', 'trend_reward_bonus']].values.tolist()
            y0 = (-log_df['final_portfolio_value']).values.tolist()
            print(f"Loaded {len(x0)} previous trials to warm-start the optimizer.")
        else:
            print("Log file is empty. Starting a new optimization.")

    # Number of new trials to run
    N_CALLS = 50

    # Prepare the arguments for the optimizer
    optimizer_kwargs = {
        "func": objective_function,
        "dimensions": SEARCH_SPACE,
        "n_calls": N_CALLS,
        "random_state": 123,
        "verbose": True
    }

    # IMPORTANT: Only add x0 and y0 if they contain data
    if x0:
        optimizer_kwargs['x0'] = x0
        optimizer_kwargs['y0'] = y0

    # Run the optimization by unpacking the keyword arguments
    result = gp_minimize(**optimizer_kwargs)

    print("\n--- Optimization Complete ---")
    print(f"Best parameters found:")
    print(f"  - stop_loss_atr_multiplier: {result.x[0]:.4f}")
    print(f"  - trend_reward_bonus: {result.x[1]:.4f}")
    print(f"Best Final Portfolio Value: ${-result.fun:,.2f}")