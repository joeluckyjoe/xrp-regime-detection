import pandas as pd
import os
import shutil
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

# Import the necessary functions from our backtesting script
from generalist_run_backtest import fetch_and_prepare_data, calculate_walk_forward_features, run_walk_forward_analysis

# --- 1. DEFINE SEARCH SPACE ---
SEARCH_SPACE = [
    Real(low=1e-5, high=1e-3, prior='log-uniform', name='lr'),
    Real(low=0.0, high=0.1, name='entropy_coef')
]
LOG_FILE = 'optimization_log_generalist.csv'
CHART_DIR = "optimization_charts" # Directory to save charts

# --- 2. PREPARE DATA ONCE ---
print("Loading and preparing all historical data for the optimization run...")
all_data_dfs = fetch_and_prepare_data(ticker="QQQ", period="729d", interval="1h")
if all_data_dfs:
    all_features, aligned_df = calculate_walk_forward_features(all_data_dfs)
    print("\nData preparation complete. Ready to start optimization.")
else:
    print("\nCRITICAL: Could not fetch data. Exiting.")
    exit()

# --- 3. DEFINE OBJECTIVE FUNCTION ---

# Global variables to track progress
RUN_COUNTER = 0
BEST_SCORE = -float('inf')

@use_named_args(SEARCH_SPACE)
def objective_function(lr, entropy_coef):
    global RUN_COUNTER, BEST_SCORE
    RUN_COUNTER += 1
    
    print("\n----------------------------------------------------")
    print(f"--- RUN {RUN_COUNTER} ---")
    print(f"Testing Parameters: lr={lr:.6f}, entropy_coef={entropy_coef:.4f}")

    params = {'lr': lr, 'entropy_coef': entropy_coef}

    # --- KEY CHANGE: Generate unique filename for this run's chart ---
    run_chart_filename = os.path.join(CHART_DIR, f"run_{RUN_COUNTER:03d}_lr_{lr:.6f}_ent_{entropy_coef:.4f}.png")
    
    results_df = run_walk_forward_analysis(
        aligned_df, all_features, total_training_steps=50000, 
        params=params, save_plot_filename=run_chart_filename
    )
    
    final_value = results_df['PortfolioValue'].iloc[-1] if not results_df.empty else 0

    print(f"\n===> Result for Run {RUN_COUNTER}: Final Portfolio Value = ${final_value:,.2f}")
    
    # --- KEY CHANGE: Check if this is the new champion ---
    if final_value > BEST_SCORE:
        BEST_SCORE = final_value
        print(f"*** New Best Score! Saving champion chart. ***")
        champion_chart_filename = os.path.join(CHART_DIR, "CHAMPION_equity_curve.png")
        shutil.copy(run_chart_filename, champion_chart_filename)

    print("----------------------------------------------------")

    with open(LOG_FILE, 'a') as f:
        f.write(f"{lr},{entropy_coef},{final_value}\n")
    
    return -final_value

# --- 4. RUN THE OPTIMIZER ---
if __name__ == '__main__':
    # --- KEY CHANGE: Create directory for charts ---
    os.makedirs(CHART_DIR, exist_ok=True)

    x0, y0 = [], []
    print("\n--- Bayesian Optimization for Generalist Champion ---")
    if not os.path.exists(LOG_FILE):
        print("Log file not found. Starting a new optimization.")
        with open(LOG_FILE, 'w') as f:
            f.write("learning_rate,entropy_coef,final_portfolio_value\n")
    else:
        print(f"Log file '{LOG_FILE}' found. Attempting to resume optimization.")
        log_df = pd.read_csv(LOG_FILE)
        if not log_df.empty:
            x0 = log_df[['learning_rate', 'entropy_coef']].values.tolist()
            y0 = (-log_df['final_portfolio_value']).values.tolist()
            # Initialize best score from the log file
            BEST_SCORE = log_df['final_portfolio_value'].max()
            print(f"Loaded {len(x0)} previous trials. Current best score: ${BEST_SCORE:,.2f}")
        else:
            print("Log file is empty. Starting a new optimization.")

    N_CALLS = 50 

    optimizer_kwargs = {"func": objective_function, "dimensions": SEARCH_SPACE, "n_calls": N_CALLS, "random_state": 123}
    if x0:
        optimizer_kwargs['x0'] = x0
        optimizer_kwargs['y0'] = y0

    result = gp_minimize(**optimizer_kwargs)

    print("\n--- Optimization Complete ---")
    print(f"Best parameters found:")
    print(f"  - learning_rate: {result.x[0]:.6f}")
    print(f"  - entropy_coef: {result.x[1]:.4f}")
    print(f"Best Final Portfolio Value: ${-result.fun:,.2f}")