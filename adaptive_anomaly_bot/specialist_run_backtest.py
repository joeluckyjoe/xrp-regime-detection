import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from collections import deque
import itertools
import warnings
import random
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.patches as mpatches

from specialist_train_rl_agent import TradingEnvironment, PPOAgent, fetch_and_prepare_data, BayesianAnomalyDetector

warnings.filterwarnings("ignore", category=RuntimeWarning, module='scipy.stats._continuous_distns')
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn.cluster._kmeans')

def set_seeds(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def create_window_fingerprint(window_df):
    if window_df.empty: return np.array([0, 0, 0])
    daily_returns = window_df['Close'].pct_change().dropna()
    if daily_returns.empty: return np.array([0, 0, 0])
    volatility = daily_returns.std()
    log_price = np.log(window_df['Close'].replace(0, 1e-9))
    x = np.arange(len(log_price))
    A = np.vstack([x, np.ones(len(x))]).T
    slope, _ = np.linalg.lstsq(A, log_price, rcond=None)[0]
    log_volume = np.log(window_df['Volume'].replace(0, 1))
    B = np.vstack([np.arange(len(log_volume)), np.ones(len(log_volume))]).T
    volume_slope, _ = np.linalg.lstsq(B, log_volume, rcond=None)[0]
    return np.array([volatility, slope, volume_slope])

def detect_regimes(historical_window_dfs, n_regimes=3, seed=42):
    if not historical_window_dfs: return None, None
    fingerprints = [create_window_fingerprint(df) for df in historical_window_dfs if not df.empty]
    if len(fingerprints) < n_regimes: return None, None
    fingerprints_arr = np.array(fingerprints)
    scaler = StandardScaler()
    scaled_fingerprints = scaler.fit_transform(fingerprints_arr)
    kmeans = KMeans(n_clusters=n_regimes, random_state=seed, n_init=10).fit(scaled_fingerprints)
    return scaler, kmeans

def map_regimes_to_labels(kmeans, scaler):
    original_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    price_momentum_col = 1
    bull_regime_index = np.argmax(original_centers[:, price_momentum_col])
    bear_regime_index = np.argmin(original_centers[:, price_momentum_col])
    regime_map = {bull_regime_index: 'BULL', bear_regime_index: 'BEAR'}
    for i in range(len(original_centers)):
        if i not in regime_map: regime_map[i] = 'SIDEWAYS'; break
    return regime_map

class RollingNormalizer:
    def __init__(self, window_size=252):
        self.window_size, self.data, self.min, self.max = window_size, deque(maxlen=window_size), None, None
    def update(self, value):
        self.data.append(value); self.min, self.max = min(self.data), max(self.data)
    def normalize(self, value):
        return 0.5 if self.max is None or self.min is None or self.max == self.min else (value - self.min) / (self.max - self.min)

def calculate_walk_forward_features(all_asset_data):
    print("Calculating v21 walk-forward multi-asset features...")
    all_features_dict = {}
    aligned_dfs_dict = {}

    for ticker, dataframes in all_asset_data.items():
        print(f"  Processing features for {ticker}...")
        timeframes=['1h','1d','1w']; indicators=['returns','Volume','rsi','atr','vix','MACD_12_26_9','MACDh_12_26_9','MACDs_12_26_9','ADX_14']
        dist_map={'returns':'t','Volume':'gamma','rsi':'beta','atr':'gamma','vix':'gamma','MACD_12_26_9':'t','MACDh_12_26_9':'t','MACDs_12_26_9':'t','ADX_14':'gamma'}
        window_sizes={'1h':882,'1d':126,'1w':26}; detectors,normalizers,data_windows={},{},{}
        
        for tf,ind in itertools.product(timeframes,indicators):
            key=f'{tf}_{ind}';win_size=window_sizes[tf]
            if win_size > len(dataframes[tf]): raise ValueError(f"window_size {win_size} for {ticker} timeframe '{tf}' is > available data ({len(dataframes[tf])}).")
            detectors[key]=BayesianAnomalyDetector(distribution_type=dist_map[ind]); normalizers[f'{key}_raw']=RollingNormalizer(win_size)
            normalizers[f'{key}_mean']=RollingNormalizer(win_size); normalizers[f'{key}_scale']=RollingNormalizer(win_size); data_windows[key]=list(dataframes[tf][ind].iloc[:win_size])
        
        for key,window in data_windows.items():
            detectors[key].fit(pd.Series(window)); [normalizers[f'{key}_raw'].update(val) for val in window]
        
        asset_features=[]; start_ts=dataframes['1w'].index[window_sizes['1w']]; start_index=dataframes['1h'].index.searchsorted(start_ts)
        
        for i in tqdm(range(start_index,len(dataframes['1h'])), desc=f"    Calculating {ticker} Features"):
            step_features=[]; current_ts=dataframes['1h'].index[i]
            for tf in timeframes:
                try:
                    tf_loc=dataframes[tf].index.searchsorted(current_ts,side='right')-1; current_observation=dataframes[tf].iloc[tf_loc]
                except(KeyError,IndexError):
                    if asset_features:step_features=asset_features[-1]; break
                for ind in indicators:
                    key=f'{tf}_{ind}';current_value=current_observation[ind]
                    if current_ts>=dataframes[tf].index[tf_loc] and len(data_windows[key])<=tf_loc:
                        data_windows[key].append(current_value)
                        if len(data_windows[key])>window_sizes[tf]: data_windows[key].pop(0)
                        detectors[key].fit(pd.Series(data_windows[key]))
                    surprise=detectors[key].compute_surprise(current_value); mean,scale=detectors[key].get_distribution_params()
                    normalizers[f'{key}_raw'].update(current_value); norm_raw=normalizers[f'{key}_raw'].normalize(current_value)
                    normalizers[f'{key}_mean'].update(mean); norm_mean=normalizers[f'{key}_mean'].normalize(mean)
                    normalizers[f'{key}_scale'].update(scale); norm_scale=normalizers[f'{key}_scale'].normalize(scale)
                    step_features.extend([surprise,norm_raw,norm_mean,norm_scale])
            
            if len(step_features)==108: asset_features.append(step_features)
            elif not step_features and asset_features: asset_features.append(asset_features[-1])

        all_features_dict[ticker] = np.array(asset_features,dtype=np.float32)
        aligned_dfs_dict[ticker] = dataframes['1h'].iloc[start_index:start_index+len(asset_features)]

    print("--- Multi-Asset Feature Calculation Complete ---")
    return all_features_dict, aligned_dfs_dict

def train_agent_on_window(train_df, train_features, total_timesteps, params, pretrained_policy=None, seed=42, reward_scheme='sortino'):
    env = TradingEnvironment(df=train_df, surprise_scores=train_features, reward_scheme=reward_scheme)
    agent = PPOAgent(env, lr=params.get('lr', 3e-4), entropy_coef=params.get('entropy_coef', 0.01))
    if pretrained_policy:
        agent.policy.load_state_dict(pretrained_policy.state_dict())
    state, _ = env.reset(seed=seed)
    
    pbar = tqdm(total=total_timesteps, desc=f"    Training ({reward_scheme})", leave=False)
    for timestep in range(total_timesteps):
        old_state, action, log_prob, value = agent.select_action(state)
        next_state, reward, done, _, _ = env.step(action)
        agent.buffer.states.append(old_state); agent.buffer.actions.append(action); agent.buffer.logprobs.append(log_prob)
        agent.buffer.rewards.append(reward); agent.buffer.is_terminals.append(done); agent.buffer.values.append(value)
        state = next_state
        if done:
            agent.learn()
            state, _ = env.reset(seed=seed + timestep + 1)
        pbar.update(1)
    pbar.close()
    return agent.policy

def test_agent_on_window(policy_net, test_df, test_features, initial_balance, mapped_label, seed=42, reward_scheme='sortino'):
    test_env = TradingEnvironment(df=test_df, surprise_scores=test_features, initial_balance=initial_balance, reward_scheme=reward_scheme)
    state, _ = test_env.reset(seed=seed)
    log = [{'Timestamp': test_df.index[0] - pd.Timedelta(hours=1), 'PortfolioValue': initial_balance, 'AssetPrice': test_df.iloc[0]['Open'], 'Regime': mapped_label}]
    done, cs = False, 0
    
    with tqdm(total=len(test_df), desc="    Testing ", leave=False) as pbar:
        while not done:
            st = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                ap, mn, _, _ = policy_net(st)
                da = torch.argmax(ap).item()
                cat = mn
                low, high = torch.tensor(test_env.action_space[1].low), torch.tensor(test_env.action_space[1].high)
                cas = low + (cat + 1.0) * 0.5 * (high - low)
                act = (da, cas.squeeze(0).numpy())
            state, _, done, _, info = test_env.step(act)
            log.append({'Timestamp': test_df.index[cs], 'PortfolioValue': info['portfolio_value'], 'AssetPrice': test_df.iloc[cs]['Close'], 'Regime': mapped_label})
            cs += 1
            pbar.update(1)
    return pd.DataFrame(log)

def run_portfolio_walk_forward(tickers, features_dict, dfs_dict, total_training_steps, params, initial_portfolio_balance=100000, save_plot_filename_base='portfolio_performance', seed=42):
    train_ws, test_ws, hist_lb, n_reg = 63*7, 21*7, 4, 3
    
    num_assets = len(tickers)
    initial_bot_balance = initial_portfolio_balance / num_assets
    current_balances = {ticker: initial_bot_balance for ticker in tickers}
    portfolio_policies = {ticker: {'BULL': None, 'BEAR': None, 'SIDEWAYS': None} for ticker in tickers}
    all_portfolio_results = []

    min_len = min(len(features) for features in features_dict.values())
    num_w = (min_len - train_ws) // test_ws

    print(f"\n--- Starting Multi-Asset Portfolio Backtest ({num_w} windows) ---")
    print(f"--- Assets: {', '.join(tickers)} ---")
    print(f"--- Initial Portfolio Balance: ${initial_portfolio_balance:,.2f} (${initial_bot_balance:,.2f} per bot) ---")

    for i in range(hist_lb, num_w):
        window_num = i + 1 - hist_lb
        print(f"\n--- Window {window_num}/{num_w - hist_lb} ---")
        window_results = []

        for ticker in tickers:
            print(f"  -- Processing Asset: {ticker} --")
            full_df = dfs_dict[ticker]
            all_features = features_dict[ticker]
            
            hist_w = []
            for j in range(i - hist_lb, i): hist_w.append(full_df.iloc[j * test_ws : j * test_ws + train_ws])
            scaler, kmeans = detect_regimes(hist_w, n_regimes=n_reg, seed=seed)
            
            if kmeans is None:
                print(f"  Skipping {ticker} in window due to insufficient data for clustering.")
                test_si, test_ei = (i * test_ws) + train_ws, (i * test_ws) + train_ws + test_ws
                if test_ei > len(full_df): continue
                test_df = full_df.iloc[test_si:test_ei]
                flat_df = pd.DataFrame({'Timestamp': test_df.index, 'PortfolioValue': current_balances[ticker]}).set_index('Timestamp')
                window_results.append(flat_df.rename(columns={'PortfolioValue': ticker}))
                continue
            
            reg_map = map_regimes_to_labels(kmeans, scaler)
            train_si, train_ei = i * test_ws, i * test_ws + train_ws
            train_df, train_feat = full_df.iloc[train_si:train_ei], all_features[train_si:train_ei]
            
            curr_fp = create_window_fingerprint(train_df); scaled_fp = scaler.transform(curr_fp.reshape(1, -1))
            curr_reg_int = kmeans.predict(scaled_fp)[0]; mapped_label = reg_map[curr_reg_int]
            rew_sch = 'sharpe' if mapped_label == 'BULL' else 'calmar'
            print(f"  Regime for {ticker}: '{mapped_label}'. Using '{rew_sch}' reward.")
            
            active_specialist_policy = portfolio_policies[ticker][mapped_label]
            current_params = params.get(mapped_label, params)
            trained_policy = train_agent_on_window(train_df, train_feat, total_training_steps, current_params, active_specialist_policy, seed + i, rew_sch)
            portfolio_policies[ticker][mapped_label] = trained_policy
            
            test_si, test_ei = train_ei, train_ei + test_ws
            if test_ei > len(all_features): continue
            test_df, test_feat = full_df.iloc[test_si:test_ei], all_features[test_si:test_ei]
            
            bot_results_df = test_agent_on_window(trained_policy, test_df, test_feat, current_balances[ticker], mapped_label, seed + i, rew_sch)
            
            if not bot_results_df.empty:
                current_balances[ticker] = bot_results_df['PortfolioValue'].iloc[-1]
                print(f"  ==> End of Window ({ticker}): Bot Value = ${current_balances[ticker]:,.2f}")
                bot_window_df = bot_results_df.set_index('Timestamp')[['PortfolioValue']].rename(columns={'PortfolioValue': ticker})
                window_results.append(bot_window_df)
        
        if window_results:
            combined_window_df = pd.concat(window_results, axis=1).ffill().bfill()
            all_portfolio_results.append(combined_window_df)
            
    print("\n--- Portfolio Walk-Forward Backtest Complete ---")
    if not all_portfolio_results: return pd.DataFrame()

    final_results_df = pd.concat(all_portfolio_results)
    final_results_df['TotalPortfolioValue'] = final_results_df.sum(axis=1)
    
    start_row = pd.DataFrame({'TotalPortfolioValue': initial_portfolio_balance}, index=[final_results_df.index[0] - pd.Timedelta(hours=1)])
    # Add columns for individual tickers to the start row
    for ticker in tickers:
        start_row[ticker] = initial_bot_balance
    final_results_df = pd.concat([start_row, final_results_df]).ffill()
    
    if save_plot_filename_base:
        print(f"\nSaving diagnostic and portfolio plots...")
        
        # --- PLOTTING LOGIC ---
        plt.style.use('seaborn-v0_8-whitegrid')

        # 1. Create Buy & Hold benchmark for all assets
        benchmark_df = pd.DataFrame(index=final_results_df.index)
        for ticker in tickers:
            asset_prices = dfs_dict[ticker]['Close'].reindex(final_results_df.index, method='ffill').bfill()
            initial_shares = initial_bot_balance / asset_prices.iloc[0]
            benchmark_df[ticker] = asset_prices * initial_shares
        benchmark_df['TotalPortfolioValue'] = benchmark_df.sum(axis=1)

        # 2. Generate individual diagnostic plots
        for ticker in tickers:
            fig, ax = plt.subplots(figsize=(12, 7))
            
            # Normalize and plot Agent Bot performance
            agent_norm = (final_results_df[ticker] / initial_bot_balance) * 100
            ax.plot(agent_norm, label=f'{ticker} Agent Bot', color='deepskyblue', linewidth=2)
            
            # Normalize and plot Buy & Hold for this asset
            benchmark_norm = (benchmark_df[ticker] / initial_bot_balance) * 100
            ax.plot(benchmark_norm, label=f'{ticker} Buy & Hold', color='gray', linestyle='--', linewidth=1.5)
            
            ax.legend(loc='upper left')
            ax.set_title(f'Diagnostic Performance: Agent vs. Buy & Hold for {ticker}', fontsize=16)
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("Normalized Value (Start = 100)", fontsize=12)
            fig.tight_layout()
            plt.savefig(f"{save_plot_filename_base}_{ticker}.png")
            plt.close()
            print(f"  - Saved {ticker} diagnostic plot.")

        # 3. Generate the final aggregated portfolio plot
        fig, ax = plt.subplots(figsize=(16, 8))
        
        agent_total_norm = (final_results_df['TotalPortfolioValue'] / initial_portfolio_balance) * 100
        ax.plot(agent_total_norm, label='Multi-Asset Agent Portfolio', color='dodgerblue', linewidth=2.5)
        
        benchmark_total_norm = (benchmark_df['TotalPortfolioValue'] / initial_portfolio_balance) * 100
        ax.plot(benchmark_total_norm, label='Equal-Weight Buy & Hold Portfolio', color='black', linestyle='--', linewidth=1.5)
        
        ax.legend(loc='upper left')
        ax.set_title("Total Multi-Asset 'Portfolio of Bots' Performance", fontsize=16)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Normalized Value (Start = 100)", fontsize=12)
        fig.tight_layout()
        plt.savefig(f"{save_plot_filename_base}_TOTAL.png")
        plt.close()
        print(f"  - Saved TOTAL portfolio plot.")
        print("All plots saved.")

    return final_results_df

if __name__ == '__main__':
    SEED = 42; set_seeds(SEED)
    
    PORTFOLIO_TICKERS = ["QQQ", "GLD", "TLT"]
    INITIAL_CAPITAL = 100000
    
    all_data = fetch_and_prepare_data(tickers=PORTFOLIO_TICKERS, period="729d", interval="1h")
    
    if all_data and len(all_data) == len(PORTFOLIO_TICKERS):
        features_dict, dfs_dict = calculate_walk_forward_features(all_data)
        
        champion_params = {
            'BULL':     {'lr': 0.000996, 'entropy_coef': 0.0015},
            'BEAR':     {'lr': 0.000093, 'entropy_coef': 0.0086},
            'SIDEWAYS': {'lr': 0.000004, 'entropy_coef': 0.0956}
        }
        
        portfolio_results = run_portfolio_walk_forward(
            tickers=PORTFOLIO_TICKERS,
            features_dict=features_dict,
            dfs_dict=dfs_dict,
            total_training_steps=50000,
            params=champion_params,
            initial_portfolio_balance=INITIAL_CAPITAL,
            save_plot_filename_base='portfolio_v1',
            seed=SEED
        )
        
        if not portfolio_results.empty:
            final_value = portfolio_results['TotalPortfolioValue'].iloc[-1]
            print(f"\n--- FINAL PORTFOLIO RESULT ---")
            print(f"Final Portfolio Value: ${final_value:,.2f}")
    else:
        print("\nCould not fetch data for all required assets. Exiting.")