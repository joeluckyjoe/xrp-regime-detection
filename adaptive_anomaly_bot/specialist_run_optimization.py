import pandas as pd
import os
import shutil
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

from specialist_run_backtest import fetch_and_prepare_data, calculate_walk_forward_features, run_walk_forward_analysis, set_seeds

# v21: Adjusted the search space for the Sideways specialist's learning rate
# based on the insight from the v20 optimization log.
SEARCH_SPACE = [
    Real(low=1e-5, high=1e-3, prior='log-uniform', name='lr_bull'),
    Real(low=0.0, high=0.1, name='entropy_coef_bull'),
    Real(low=1e-5, high=1e-3, prior='log-uniform', name='lr_bear'),
    Real(low=0.0, high=0.1, name='entropy_coef_bear'),
    Real(low=1e-6, high=1e-4, prior='log-uniform', name='lr_sideways'), # <-- This line is changed
    Real(low=0.0, high=0.1, name='entropy_coef_sideways')
]
LOG_FILE = 'optimization_log_specialist_v21_calmar.csv'
CHART_DIR = "optimization_charts_v21_calmar"
SEED = 42

print("Loading and preparing all historical data for the optimization run...")
set_seeds(SEED)
all_data_dfs = fetch_and_prepare_data(ticker="QQQ", period="729d", interval="1h")
if all_data_dfs:
    all_features, aligned_df = calculate_walk_forward_features(all_data_dfs)
    print("\nData preparation complete. Ready to start optimization.")
else:
    print("\nCRITICAL: Could not fetch data. Exiting.")
    exit()

RUN_COUNTER = 0; BEST_SCORE = -float('inf')

@use_named_args(SEARCH_SPACE)
def objective_function(lr_bull, entropy_coef_bull, lr_bear, entropy_coef_bear, lr_sideways, entropy_coef_sideways):
    global RUN_COUNTER, BEST_SCORE
    RUN_COUNTER += 1
    print("\n----------------------------------------------------")
    print(f"--- RUN {RUN_COUNTER} ---")
    params = {
        'BULL': {'lr': lr_bull, 'entropy_coef': entropy_coef_bull},
        'BEAR': {'lr': lr_bear, 'entropy_coef': entropy_coef_bear},
        'SIDEWAYS': {'lr': lr_sideways, 'entropy_coef': entropy_coef_sideways}
    }
    print("Testing Parameters:")
    for regime, p in params.items(): print(f"  - {regime}: lr={p['lr']:.6f}, entropy_coef={p['entropy_coef']:.4f}")
    run_chart_filename = os.path.join(CHART_DIR, f"run_{RUN_COUNTER:03d}.png")
    results_df = run_walk_forward_analysis(aligned_df, all_features, total_training_steps=50000, params=params, save_plot_filename=run_chart_filename, seed=SEED)
    final_value = results_df['PortfolioValue'].iloc[-1] if not results_df.empty else 0
    print(f"\n===> Result for Run {RUN_COUNTER}: Final Portfolio Value = ${final_value:,.2f}")
    if final_value > BEST_SCORE:
        BEST_SCORE = final_value
        print(f"*** New Best Score! Saving champion chart. ***")
        shutil.copy(run_chart_filename, os.path.join(CHART_DIR, "CHAMPION_equity_curve.png"))
    print("----------------------------------------------------")
    with open(LOG_FILE, 'a') as f:
        f.write(f"{lr_bull},{entropy_coef_bull},{lr_bear},{entropy_coef_bear},{lr_sideways},{entropy_coef_sideways},{final_value}\n")
    return -final_value

if __name__ == '__main__':
    os.makedirs(CHART_DIR, exist_ok=True)
    print("\n--- Bayesian Optimization for Specialist Champions v21 (Calmar) ---")
    if not os.path.exists(LOG_FILE):
        print("Log file not found. Starting a new optimization.")
        with open(LOG_FILE, 'w') as f:
            f.write("lr_bull,entropy_coef_bull,lr_bear,entropy_coef_bear,lr_sideways,entropy_coef_sideways,final_portfolio_value\n")
        x0, y0 = None, None
    else:
        print(f"Log file '{LOG_FILE}' found. Attempting to resume optimization.")
        log_df = pd.read_csv(LOG_FILE)
        if not log_df.empty:
            param_cols = ['lr_bull','entropy_coef_bull','lr_bear','entropy_coef_bear','lr_sideways','entropy_coef_sideways']
            x0 = log_df[param_cols].values.tolist(); y0 = (-log_df['final_portfolio_value']).values.tolist()
            BEST_SCORE = log_df['final_portfolio_value'].max()
            print(f"Loaded {len(x0)} previous trials. Current best score: ${BEST_SCORE:,.2f}")
        else:
            print("Log file is empty. Starting a new optimization."); x0, y0 = None, None
    N_CALLS = 50
    optimizer_kwargs = {"func": objective_function, "dimensions": SEARCH_SPACE, "n_calls": N_CALLS, "random_state": 123}
    if x0 and y0: optimizer_kwargs['x0'], optimizer_kwargs['y0'] = x0, y0
    result = gp_minimize(**optimizer_kwargs)
    print("\n--- Optimization Complete ---")
    print(f"Best parameters found:")
    best_params = {p.name: val for p, val in zip(SEARCH_SPACE, result.x)}
    print(f"  - BULL:     lr={best_params['lr_bull']:.6f}, entropy_coef={best_params['entropy_coef_bull']:.4f}")
    print(f"  - BEAR:     lr={best_params['lr_bear']:.6f}, entropy_coef={best_params['entropy_coef_bear']:.4f}")
    print(f"  - SIDEWAYS: lr={best_params['lr_sideways']:.6f}, entropy_coef={best_params['entropy_coef_sideways']:.4f}")
    print(f"Best Final Portfolio Value: ${-result.fun:,.2f}")