import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from collections import deque
import itertools
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning, module='scipy.stats._continuous_distns')
from generalist_train_rl_agent import TradingEnvironment, PPOAgent, fetch_and_prepare_data, BayesianAnomalyDetector

class RollingNormalizer:
    # ... (code is unchanged) ...
    def __init__(self, window_size=252):
        self.window_size = window_size
        self.data = deque(maxlen=window_size)
        self.min, self.max = None, None
    def update(self, value):
        self.data.append(value)
        self.min, self.max = min(self.data), max(self.data)
    def normalize(self, value):
        if self.max is None or self.min is None or self.max == self.min: return 0.5
        return (value - self.min) / (self.max - self.min)

def calculate_walk_forward_features(dataframes):
    print("Calculating v18 walk-forward multi-scale features (108 per step)...")
    
    timeframes = ['1h', '1d', '1w']
    # --- v18: EXPANDED INDICATOR LIST ---
    indicators = [
        'returns', 'Volume', 'rsi', 'atr', 'vix',
        'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'ADX_14'
    ]
    # --- v18: EXPANDED DISTRIBUTION MAP ---
    dist_map = {
        'returns':'t', 'Volume':'gamma', 'rsi':'beta', 'atr':'gamma', 'vix':'gamma',
        'MACD_12_26_9':'t', 'MACDh_12_26_9':'t', 'MACDs_12_26_9':'t', 'ADX_14':'gamma'
    }
    
    window_sizes = {'1h': 882, '1d': 126, '1w': 26}
    detectors, normalizers, data_windows = {}, {}, {}
    for tf, ind in itertools.product(timeframes, indicators):
        key = f'{tf}_{ind}'; win_size = window_sizes[tf]
        if win_size > len(dataframes[tf]):
            raise ValueError(f"window_size {win_size} for timeframe '{tf}' is > available data ({len(dataframes[tf])}).")
        detectors[key] = BayesianAnomalyDetector(distribution_type=dist_map[ind])
        normalizers[f'{key}_raw']=RollingNormalizer(win_size)
        normalizers[f'{key}_mean']=RollingNormalizer(win_size)
        normalizers[f'{key}_scale']=RollingNormalizer(win_size)
        data_windows[key] = list(dataframes[tf][ind].iloc[:win_size])

    for key, window in data_windows.items():
        detectors[key].fit(pd.Series(window))
        for val in window:
            normalizers[f'{key}_raw'].update(val)

    all_features = []
    start_ts = dataframes['1w'].index[window_sizes['1w']]
    start_index = dataframes['1h'].index.searchsorted(start_ts)

    for i in tqdm(range(start_index, len(dataframes['1h']))):
        step_features = []
        current_ts = dataframes['1h'].index[i]
        for tf in timeframes:
            try:
                tf_loc = dataframes[tf].index.searchsorted(current_ts, side='right')-1
                current_observation = dataframes[tf].iloc[tf_loc]
            except (KeyError, IndexError):
                if all_features: step_features = all_features[-1]
                break 
            for ind in indicators:
                key=f'{tf}_{ind}'; current_value=current_observation[ind]
                if current_ts >= dataframes[tf].index[tf_loc] and len(data_windows[key]) <= tf_loc:
                    data_windows[key].append(current_value)
                    win_size = window_sizes[tf]
                    if len(data_windows[key]) > win_size:
                        data_windows[key].pop(0)
                    detectors[key].fit(pd.Series(data_windows[key]))
                surprise=detectors[key].compute_surprise(current_value)
                mean,scale=detectors[key].get_distribution_params()
                normalizers[f'{key}_raw'].update(current_value)
                norm_raw=normalizers[f'{key}_raw'].normalize(current_value)
                normalizers[f'{key}_mean'].update(mean)
                norm_mean=normalizers[f'{key}_mean'].normalize(mean)
                normalizers[f'{key}_scale'].update(scale)
                norm_scale=normalizers[f'{key}_scale'].normalize(scale)
                step_features.extend([surprise,norm_raw,norm_mean,norm_scale])
        
        # --- v18: UPDATE STATE SIZE CHECK ---
        if len(step_features) == 108:
             all_features.append(step_features)
        elif not step_features and all_features:
             all_features.append(all_features[-1])

    return np.array(all_features, dtype=np.float32), dataframes['1h'].iloc[start_index:start_index+len(all_features)]

# ... (train_agent_on_window, test_agent_on_window, and run_walk_forward_analysis are unchanged) ...
def train_agent_on_window(train_df, train_features, total_timesteps, params={}):
    env = TradingEnvironment(df=train_df, surprise_scores=train_features)
    agent = PPOAgent(env, lr=params.get('lr', 3e-4), entropy_coef=params.get('entropy_coef', 0.01))
    state, _ = env.reset()
    for timestep in tqdm(range(total_timesteps), desc="  Training"):
        old_state, action, log_prob, value = agent.select_action(state)
        next_state, reward, done, _, _ = env.step(action)
        agent.buffer.states.append(old_state); agent.buffer.actions.append(action)
        agent.buffer.logprobs.append(log_prob); agent.buffer.rewards.append(reward)
        agent.buffer.is_terminals.append(done); agent.buffer.values.append(value)
        state = next_state
        if done:
            agent.learn()
            state, _ = env.reset()
    return agent.policy
def test_agent_on_window(policy_net, test_df, test_features, initial_balance):
    test_env = TradingEnvironment(df=test_df, surprise_scores=test_features, initial_balance=initial_balance)
    state, _ = test_env.reset()
    results_log = [{'Timestamp': test_df.index[0] - pd.Timedelta(hours=1), 'PortfolioValue': initial_balance, 'AssetPrice': test_df.iloc[0]['Open']}]
    done = False; current_step = 0
    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_probs, means, _, _ = policy_net(state_tensor)
            discrete_action = torch.argmax(action_probs).item()
            continuous_actions_tanh = means
            low=torch.tensor(test_env.action_space[1].low); high=torch.tensor(test_env.action_space[1].high)
            continuous_actions_scaled = low + (continuous_actions_tanh + 1.0) * 0.5 * (high - low)
            action = (discrete_action, continuous_actions_scaled.squeeze(0).numpy())
        state, _, done, _, info = test_env.step(action)
        results_log.append({'Timestamp': test_df.index[current_step], 'PortfolioValue': info['portfolio_value'], 'AssetPrice': test_df.iloc[current_step]['Close']})
        current_step += 1
    return pd.DataFrame(results_log)
def run_walk_forward_analysis(full_df, all_features, total_training_steps, params={}, save_plot_filename=None):
    train_window_size, test_window_size = 63 * 7, 21 * 7 
    initial_balance, current_balance = 100000, 100000
    all_results_df = []
    num_windows = (len(all_features) - train_window_size) // test_window_size
    print(f"\n--- Starting Walk-Forward Backtest ({num_windows} windows) ---")
    for i in range(num_windows):
        train_start_idx = i*test_window_size; train_end_idx = train_start_idx+train_window_size
        test_start_idx, test_end_idx = train_end_idx, train_end_idx+test_window_size
        if test_end_idx > len(all_features):
            print(f"\nSkipping final window as it is incomplete.")
            break
        train_df_slice = full_df.iloc[train_start_idx:train_end_idx]
        train_features_slice = all_features[train_start_idx:train_end_idx]
        test_df_slice = full_df.iloc[test_start_idx:test_end_idx]
        test_features_slice = all_features[test_start_idx:test_end_idx]
        print(f"\n[Window {i+1}/{num_windows}] Training agent...")
        trained_policy_net = train_agent_on_window(train_df_slice, train_features_slice, total_training_steps, params)
        print(f"[Window {i+1}/{num_windows}] Testing agent...")
        monthly_results_df = test_agent_on_window(trained_policy_net, test_df_slice, test_features_slice, current_balance)
        all_results_df.append(monthly_results_df.iloc[1:])
        if not monthly_results_df.empty:
            current_balance = monthly_results_df['PortfolioValue'].iloc[-1]
            end_date = monthly_results_df['Timestamp'].iloc[-1].strftime('%Y-%m-%d')
            print(f"  ==> End of Window {i+1} ({end_date}): Portfolio Value = ${current_balance:,.2f}")
    print("\n--- Walk-Forward Backtest Complete ---")
    if not all_results_df: return pd.DataFrame()
    final_results = pd.concat(all_results_df)
    start_row = pd.DataFrame([{'Timestamp': final_results['Timestamp'].iloc[0] - pd.Timedelta(hours=1), 'PortfolioValue': initial_balance, 'AssetPrice': final_results['AssetPrice'].iloc[0]}])
    results_df = pd.concat([start_row, final_results]).set_index('Timestamp')
    if save_plot_filename and not results_df.empty:
        print(f"\nSaving equity curve plot to '{save_plot_filename}'...")
        plt.figure(figsize=(14, 7))
        normalized_portfolio = (results_df['PortfolioValue'] / results_df['PortfolioValue'].iloc[0]) * 100
        normalized_asset = (results_df['AssetPrice'] / results_df['AssetPrice'].iloc[0]) * 100
        plt.plot(normalized_portfolio, label='Agent Equity Curve', color='deepskyblue')
        plt.plot(normalized_asset, label='QQQ Asset Price (Buy & Hold)', color='gray', linestyle='--')
        plt.title("Agent Performance vs. Buy & Hold")
        plt.xlabel("Date"); plt.ylabel("Normalized Value (Start = 100)")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5); plt.legend(); plt.tight_layout()
        plt.savefig(save_plot_filename)
        plt.close()
        print("Plot saved successfully.")
    return results_df

if __name__ == '__main__':
    all_data_dfs = fetch_and_prepare_data(ticker="QQQ", period="729d", interval="1h")
    if all_data_dfs:
        all_features, aligned_df = calculate_walk_forward_features(all_data_dfs)
        results_df = run_walk_forward_analysis(
            aligned_df, all_features, total_training_steps=50000,
            # --- v18: UPDATE FILENAME FOR BASELINE RUN ---
            save_plot_filename='equity_curve_vs_asset_v18_baseline.png'
        )
        if not results_df.empty:
            print(f"Final Portfolio Value: ${results_df['PortfolioValue'].iloc[-1]:,.2f}")