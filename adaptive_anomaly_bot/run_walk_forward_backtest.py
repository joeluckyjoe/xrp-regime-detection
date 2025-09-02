import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from collections import deque

from train_rl_agent import TradingEnvironment, Agent, fetch_and_prepare_data, BayesianAnomalyDetector

class RollingNormalizer:
    def __init__(self, window_size=252):
        self.window_size = window_size
        self.data = deque(maxlen=window_size)
        self.min = None
        self.max = None
    def update(self, value):
        self.data.append(value)
        self.min = min(self.data)
        self.max = max(self.data)
    def normalize(self, value):
        if self.max is None or self.min is None or self.max == self.min: return 0.5
        return (value - self.min) / (self.max - self.min)

def calculate_walk_forward_features(full_df):
    print("Calculating walk-forward features (28 per step) with VIX integration...")
    dist_map = {
        'Volume': 'gamma', 'returns': 't', 'rsi': 'beta',
        'atr': 'gamma', 'sma_dist': 'gamma', 'vol_regime': 'gamma',
        'vix': 'gamma'
    }
    features_to_model = list(dist_map.keys())
    window_size = 441
    detectors = {f: BayesianAnomalyDetector(distribution_type=dist_map[f]) for f in features_to_model}
    raw_val_normalizers = {f: RollingNormalizer(window_size) for f in features_to_model}
    mean_normalizers = {f: RollingNormalizer(window_size) for f in features_to_model}
    scale_normalizers = {f: RollingNormalizer(window_size) for f in features_to_model}
    data_windows = {f: list(full_df[f].iloc[:window_size]) for f in features_to_model}
    all_features = []

    for feature in features_to_model:
        detectors[feature].fit(pd.Series(data_windows[feature]))
        for val in data_windows[feature]:
            raw_val_normalizers[feature].update(val)

    for i in tqdm(range(window_size, len(full_df))):
        current_observation = full_df.iloc[i]
        step_features = []
        for feature in features_to_model:
            data_windows[feature].append(current_observation[feature])
            if len(data_windows[feature]) > window_size:
                data_windows[feature].pop(0)
            detectors[feature].fit(pd.Series(data_windows[feature]))
            surprise_score = detectors[feature].compute_surprise(current_observation[feature])
            mean_param, scale_param = detectors[feature].get_distribution_params()
            raw_val_normalizers[feature].update(current_observation[feature])
            norm_raw_val = raw_val_normalizers[feature].normalize(current_observation[feature])
            mean_normalizers[feature].update(mean_param)
            norm_mean = mean_normalizers[feature].normalize(mean_param)
            scale_normalizers[feature].update(scale_param)
            norm_scale = scale_normalizers[feature].normalize(scale_param)
            step_features.extend([surprise_score, norm_raw_val, norm_mean, norm_scale])
        all_features.append(step_features)
    return np.array(all_features, dtype=np.float32)

def train_agent_on_window(train_df, train_features, num_episodes, params):
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
            if terminated: done = True
    return agent.policy_net

def test_agent_on_window(policy_net, test_df, test_features, initial_balance, params):
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

def run_walk_forward_analysis(full_df, all_features, num_episodes, params):
    # Original hourly window sizes
    train_window_size, test_window_size = 63 * 7, 21 * 7
    initial_balance, current_balance = 100000, 100000
    all_results_df = []
    num_windows = (len(all_features) - train_window_size) // test_window_size
    print(f"\n--- Starting Walk-Forward Backtest ({num_windows} windows) ---")

    for i in tqdm(range(num_windows)):
        # Define all indices first
        train_start_idx = i * test_window_size
        train_end_idx = train_start_idx + train_window_size
        test_start_idx = train_end_idx
        test_end_idx = test_start_idx + test_window_size

        # CORRECTED: Now perform the boundary check *after* test_end_idx is defined
        if test_end_idx > len(all_features):
            print(f"\nSkipping final window as it is incomplete.")
            break

        # Slice data now that indices are validated
        train_df_slice = full_df.iloc[train_start_idx:train_end_idx]
        train_features_slice = all_features[train_start_idx:train_end_idx]
        test_df_slice = full_df.iloc[test_start_idx:test_end_idx]
        test_features_slice = all_features[test_start_idx:test_end_idx]

        print(f"\n[Window {i+1}/{num_windows}] Training agent...")
        trained_policy_net = train_agent_on_window(train_df_slice, train_features_slice, num_episodes, params)
        
        print(f"[Window {i+1}/{num_windows}] Testing agent...")
        monthly_results_df = test_agent_on_window(trained_policy_net, test_df_slice, test_features_slice, current_balance, params)
        
        all_results_df.append(monthly_results_df.iloc[1:])
        if not monthly_results_df.empty:
            current_balance = monthly_results_df['PortfolioValue'].iloc[-1]
            end_date = monthly_results_df['Timestamp'].iloc[-1].strftime('%Y-%m-%d')
            print(f"  ==> End of Window {i+1} ({end_date}): Portfolio Value = ${current_balance:,.2f}")

    print("\n--- Walk-Forward Backtest Complete ---")
    if not all_results_df: return pd.DataFrame()
    final_results = pd.concat(all_results_df)
    start_row = pd.DataFrame([{'Timestamp': final_results['Timestamp'].iloc[0] - pd.Timedelta(hours=1), 'PortfolioValue': initial_balance, 'AssetPrice': final_results['AssetPrice'].iloc[0]}])
    return pd.concat([start_row, final_results]).set_index('Timestamp')

if __name__ == '__main__':
    # --- Define the Parameter Sets for Sensitivity Analysis ---
    V11_CHAMPION = { 'trade_penalty': 0.0954, 'trend_reward_bonus': 0.0376, 'stop_loss_atr_multiplier': 1.50 }

    sensitivity_params = {
        "Champion": V11_CHAMPION,
        "Penalty -10%": {**V11_CHAMPION, 'trade_penalty': 0.085},
        "Penalty +10%": {**V11_CHAMPION, 'trade_penalty': 0.105},
        "Bonus -10%": {**V11_CHAMPION, 'trend_reward_bonus': 0.033},
        "Bonus +10%": {**V11_CHAMPION, 'trend_reward_bonus': 0.041},
        "Stop-Loss -10%": {**V11_CHAMPION, 'stop_loss_atr_multiplier': 1.35},
        "Stop-Loss +10%": {**V11_CHAMPION, 'stop_loss_atr_multiplier': 1.65},
    }

    print("--- INITIATING ROBUSTNESS TEST 3: PARAMETER SENSITIVITY ANALYSIS ---")
    print("Loading recent hourly historical data for QQQ...")
    full_data_df = fetch_and_prepare_data(ticker="QQQ", period="729d", interval="1h")

    if full_data_df is not None:
        feature_window_size = 441
        print("Calculating walk-forward features (this will only be done once)...")
        all_features = calculate_walk_forward_features(full_data_df)
        aligned_df = full_data_df.iloc[feature_window_size:]

        results_summary = {}

        for name, params in sensitivity_params.items():
            print(f"\n\n{'='*20} RUNNING BACKTEST FOR: {name.upper()} {'='*20}")
            # We pass the already calculated features to avoid recalculating them each time
            results_df = run_walk_forward_analysis(aligned_df, all_features, num_episodes=150, params=params)
            
            if not results_df.empty:
                initial_value = results_df['PortfolioValue'].iloc[0]
                final_value = results_df['PortfolioValue'].iloc[-1]
                percent_return = ((final_value / initial_value) - 1) * 100
                results_summary[name] = (final_value, percent_return)
                print(f"  ==> Final Value for {name}: ${final_value:,.2f} ({percent_return:+.2f}%)")

        print("\n\n--- SENSITIVITY ANALYSIS COMPLETE ---")
        print("Final Portfolio Value Summary:")
        for name, (value, ret) in results_summary.items():
            print(f"  - {name}: ${value:,.2f} ({ret:+.2f}%)")