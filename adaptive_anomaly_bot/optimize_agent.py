import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

# --- Import necessary components from our existing scripts ---
from train_rl_agent import TradingEnvironment, Agent, fetch_and_prepare_data, BayesianAnomalyDetector
from run_walk_forward_backtest import calculate_walk_forward_features

# --- Define the Hyperparameter Search Space ---
dim_trade_penalty = Real(low=0.05, high=0.50, name='trade_penalty', prior='log-uniform')
dim_trend_reward_bonus = Real(low=0.0, high=0.05, name='trend_reward_bonus', prior='uniform')
dim_stop_loss_atr = Real(low=1.5, high=3.5, name='stop_loss_atr_multiplier', prior='uniform')

SPACE = [dim_trade_penalty, dim_trend_reward_bonus, dim_stop_loss_atr]

# Global variables to track progress
ITERATION_COUNT = 0
BEST_SCORE = -np.inf

# --- Helper Functions ---

def calculate_sortino_ratio(returns):
    mean_return = returns.mean()
    negative_returns = returns[returns < 0]
    downside_deviation = negative_returns.std(ddof=0)
    if pd.isna(downside_deviation) or downside_deviation == 0:
        return 0 if mean_return <= 0 else 5.0 # Return a large finite number for risk-free positive returns
    sortino = mean_return / downside_deviation * np.sqrt(252 * 7)
    return sortino

def save_equity_plot(results_df, filename, title):
    plt.figure(figsize=(14, 7))
    normalized_portfolio = (results_df['PortfolioValue'] / results_df['PortfolioValue'].iloc[0]) * 100
    normalized_asset = (results_df['AssetPrice'] / results_df['AssetPrice'].iloc[0]) * 100
    plt.plot(normalized_portfolio, label='Agent Equity Curve', color='deepskyblue')
    plt.plot(normalized_asset, label='QQQ Asset Price (Buy & Hold)', color='gray', linestyle='--')
    plt.title(title, fontsize=12)
    plt.xlabel("Date"); plt.ylabel("Normalized Value (Start = 100)")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5); plt.legend(); plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# --- Core Backtesting Functions (Modified to accept params) ---

def train_agent_on_window(train_df, train_features, num_episodes=150, params=None):
    if params is None:
        params = {}
    env = TradingEnvironment(df=train_df, surprise_scores=train_features, **params)
    agent = Agent(env)
    for i_episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        done = False
        while not done:
            action = agent.select_action(state)
            observation, reward, terminated, _, _ = env.step(action.item())
            reward = torch.tensor([reward], dtype=torch.float32)
            next_state = None if terminated else torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
            agent.memory.push(state, action, next_state, reward)
            state = next_state
            agent.optimize_model()
            target_net_state_dict = agent.target_net.state_dict()
            policy_net_state_dict = agent.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*agent.TAU + target_net_state_dict[key]*(1-agent.TAU)
            agent.target_net.load_state_dict(target_net_state_dict)
            if terminated:
                done = True
    return agent.policy_net

def test_agent_on_window(policy_net, test_df, test_features, initial_balance, params=None):
    if params is None:
        params = {}
    test_env = TradingEnvironment(df=test_df, surprise_scores=test_features, initial_balance=initial_balance, **params)
    state, _ = test_env.reset()
    results_log = [{'Timestamp': test_df.index[0] - pd.Timedelta(hours=1), 'PortfolioValue': initial_balance, 'AssetPrice': test_df.iloc[0]['Open']}]
    done = False
    current_step = 0
    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = policy_net(state_tensor).max(1)[1].view(1, 1).item()
        state, _, done, _, info = test_env.step(action)
        results_log.append({'Timestamp': test_df.index[current_step], 'PortfolioValue': info['portfolio_value'], 'AssetPrice': test_df.iloc[current_step]['Close']})
        current_step += 1
    return pd.DataFrame(results_log)


def run_walk_forward_analysis_parameterized(full_df, all_features, params):
    train_window_size, test_window_size = 63 * 7, 21 * 7
    initial_balance, current_balance = 100000, 100000
    all_results_df = []
    num_windows = (len(all_features) - train_window_size) // test_window_size

    for i in range(num_windows):
        train_start_idx = i * test_window_size
        train_end_idx = train_start_idx + train_window_size
        test_start_idx = train_end_idx
        test_end_idx = train_end_idx + test_window_size
        
        train_df_slice = full_df.iloc[train_start_idx:train_end_idx]
        train_features_slice = all_features[train_start_idx:train_end_idx]
        
        test_df_slice = full_df.iloc[test_start_idx:test_end_idx]
        test_features_slice = all_features[test_start_idx:test_end_idx]
        
        trained_policy_net = train_agent_on_window(
            train_df=train_df_slice, 
            train_features=train_features_slice, 
            num_episodes=150,
            params=params
        )
        
        monthly_results_df = test_agent_on_window(
            policy_net=trained_policy_net,
            test_df=test_df_slice,
            test_features=test_features_slice,
            initial_balance=current_balance,
            params=params
        )

        all_results_df.append(monthly_results_df.iloc[1:])
        if not monthly_results_df.empty:
            current_balance = monthly_results_df['PortfolioValue'].iloc[-1]
            
    if not all_results_df: return pd.DataFrame()
    final_results = pd.concat(all_results_df)
    start_row = pd.DataFrame([{'Timestamp': final_results['Timestamp'].iloc[0] - pd.Timedelta(hours=1), 'PortfolioValue': initial_balance, 'AssetPrice': final_results['AssetPrice'].iloc[0]}])
    return pd.concat([start_row, final_results]).set_index('Timestamp')

# --- The Core Objective Function for the Optimizer ---

@use_named_args(dimensions=SPACE)
def objective_function(trade_penalty, trend_reward_bonus, stop_loss_atr_multiplier):
    global ITERATION_COUNT, BEST_SCORE
    ITERATION_COUNT += 1
    
    print(f"\n--- Iteration {ITERATION_COUNT}/{N_CALLS} ---")
    print(f"Testing Parameters:")
    print(f"  - trade_penalty: {trade_penalty:.4f}")
    print(f"  - trend_reward_bonus: {trend_reward_bonus:.4f}")
    print(f"  - stop_loss_atr_multiplier: {stop_loss_atr_multiplier:.4f}")

    params = {
        'trade_penalty': trade_penalty,
        'trend_reward_bonus': trend_reward_bonus,
        'stop_loss_atr_multiplier': stop_loss_atr_multiplier
    }

    results_df = run_walk_forward_analysis_parameterized(ALIGNED_DF, ALL_FEATURES, params)

    if results_df.empty or len(results_df) < 2:
        print("Backtest returned no results. Assigning worst possible score.")
        return 100.0

    portfolio_returns = results_df['PortfolioValue'].pct_change().dropna()
    score = calculate_sortino_ratio(portfolio_returns)
    
    print(f"==> Iteration {ITERATION_COUNT} Complete. Sortino Ratio: {score:.4f}")

    filename = f"optimization_plots/iter_{ITERATION_COUNT:02d}_sortino_{score:.2f}.png"
    title = f"Iter {ITERATION_COUNT} | Penalty={trade_penalty:.2f}, Bonus={trend_reward_bonus:.2f}, SL={stop_loss_atr_multiplier:.2f} | Sortino={score:.2f}"
    save_equity_plot(results_df, filename, title)
    print(f"Saved equity curve to {filename}")

    if score > BEST_SCORE:
        BEST_SCORE = score
        print(f"*** New best score found! Saving champion plot. ***")
        save_equity_plot(results_df, "best_equity_curve_so_far.png", f"Best So Far (Iter {ITERATION_COUNT}) - {title}")

    return -score


if __name__ == '__main__':
    N_CALLS = 30
    
    if not os.path.exists('optimization_plots'):
        os.makedirs('optimization_plots')
        print("Created directory 'optimization_plots' to store iteration charts.")

    print("\n--- Preparing Data (One-Time Setup) ---")
    full_data_df = fetch_and_prepare_data(period="2y")
    if full_data_df is None:
        print("Failed to fetch data. Exiting.")
        exit()
        
    feature_window_size = 441
    ALL_FEATURES = calculate_walk_forward_features(full_data_df)
    ALIGNED_DF = full_data_df.iloc[feature_window_size:]
    print("--- Data preparation complete. Starting optimization. ---")

    result = gp_minimize(
        func=objective_function,
        dimensions=SPACE,
        n_calls=N_CALLS,
        random_state=42
    )

    print("\n\n--- Bayesian Optimization Complete ---")
    print(f"Best Sortino Ratio found: {-result.fun:.4f}")
    print("Best Parameters:")
    print(f"  - trade_penalty: {result.x[0]:.4f}")
    print(f"  - trend_reward_bonus: {result.x[1]:.4f}")
    print(f"  - stop_loss_atr_multiplier: {result.x[2]:.4f}")
    print("\nThese parameters define the new champion v11 agent.")