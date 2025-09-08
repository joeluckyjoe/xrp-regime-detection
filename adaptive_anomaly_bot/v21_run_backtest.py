import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from collections import deque
import itertools
import warnings
import random
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture
import matplotlib.patches as mpatches

from v21_train_rl_agent import TradingEnvironment, PPOAgent, fetch_and_prepare_data, BayesianAnomalyDetector

warnings.filterwarnings("ignore", category=RuntimeWarning, module='scipy.stats._continuous_distns')
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn.cluster._kmeans')
warnings.filterwarnings("ignore", category=FutureWarning)


def set_seeds(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class RollingNormalizer:
    def __init__(self, window_size=252):
        self.window_size, self.data, self.min, self.max = window_size, deque(maxlen=window_size), None, None
    def update(self, value):
        self.data.append(value); self.min, self.max = min(self.data), max(self.data)
    def normalize(self, value):
        return 0.5 if self.max is None or self.min is None or self.max == self.min else (value - self.min) / (self.max - self.min)

def calculate_walk_forward_features(dataframes):
    print("Calculating v21 walk-forward multi-scale features (108 per step)...")
    timeframes=['1h','1d','1w']; indicators=['returns','Volume','rsi','atr','vix','MACD_12_26_9','MACDh_12_26_9','MACDs_12_26_9','ADX_14']
    dist_map={'returns':'t','Volume':'gamma','rsi':'beta','atr':'gamma','vix':'gamma','MACD_12_26_9':'t','MACDh_12_26_9':'t','MACDs_12_26_9':'t','ADX_14':'gamma'}
    window_sizes={'1h':882,'1d':126,'1w':26}; detectors,normalizers,data_windows={},{},{}
    for tf,ind in itertools.product(timeframes,indicators):
        key=f'{tf}_{ind}';win_size=window_sizes[tf]
        if win_size > len(dataframes[tf]): raise ValueError(f"window_size {win_size} for timeframe '{tf}' is > available data ({len(dataframes[tf])}).")
        detectors[key]=BayesianAnomalyDetector(distribution_type=dist_map[ind]); normalizers[f'{key}_raw']=RollingNormalizer(win_size)
        normalizers[f'{key}_mean']=RollingNormalizer(win_size); normalizers[f'{key}_scale']=RollingNormalizer(win_size); data_windows[key]=list(dataframes[tf][ind].iloc[:win_size])
    for key,window in data_windows.items():
        detectors[key].fit(pd.Series(window)); [normalizers[f'{key}_raw'].update(val) for val in window]
    all_features=[]; start_ts=dataframes['1w'].index[window_sizes['1w']]; start_index=dataframes['1h'].index.searchsorted(start_ts)
    for i in tqdm(range(start_index,len(dataframes['1h'])), desc="  Calculating Features"):
        step_features=[]; current_ts=dataframes['1h'].index[i]
        for tf in timeframes:
            try:
                tf_loc=dataframes[tf].index.searchsorted(current_ts,side='right')-1; current_observation=dataframes[tf].iloc[tf_loc]
            except(KeyError,IndexError):
                if all_features:step_features=all_features[-1]; break
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
        if len(step_features)==108: all_features.append(step_features)
        elif not step_features and all_features: all_features.append(all_features[-1])
    return np.array(all_features,dtype=np.float32),dataframes['1h'].iloc[start_index:start_index+len(all_features)]

def detect_regimes(historical_feature_vectors, max_regimes=8, seed=42):
    if len(historical_feature_vectors) < max_regimes * 5:
        return None, None, None, 0
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(historical_feature_vectors)
    pca = PCA(n_components=0.95, random_state=seed)
    reduced_features = pca.fit_transform(scaled_features)
    bgmm = BayesianGaussianMixture(
        n_components=max_regimes, n_init=10, weight_concentration_prior_type='dirichlet_process',
        weight_concentration_prior=0.1, random_state=seed
    )
    bgmm.fit(reduced_features)
    active_components = np.sum(bgmm.weights_ > 1.0 / max_regimes)
    if active_components == 0: active_components = 1
    return scaler, pca, bgmm, active_components

def map_regimes_to_labels(scaler, pca, bgmm, n_regimes):
    if n_regimes == 0:
        return {}, {}
    cluster_centers_reduced = bgmm.means_
    active_indices = np.argsort(bgmm.weights_)[::-1][:n_regimes]
    cluster_centers_original_scaled = pca.inverse_transform(cluster_centers_reduced[active_indices])
    cluster_centers_original = scaler.inverse_transform(cluster_centers_original_scaled)
    momentum_proxy_index = 1
    momentums = cluster_centers_original[:, momentum_proxy_index]
    sorted_momentum_indices = np.argsort(momentums)

    regime_map = {}
    if n_regimes == 1:
        regime_map[active_indices[sorted_momentum_indices[0]]] = 'NEUTRAL_1'
    else:
        bull_cluster_index = active_indices[sorted_momentum_indices[-1]]
        bear_cluster_index = active_indices[sorted_momentum_indices[0]]
        regime_map[bull_cluster_index] = 'BULL'
        regime_map[bear_cluster_index] = 'BEAR'
        neutral_count = 1
        for i in range(1, n_regimes - 1):
            neutral_cluster_index = active_indices[sorted_momentum_indices[i]]
            regime_map[neutral_cluster_index] = f'NEUTRAL_{neutral_count}'
            neutral_count += 1
    return regime_map, {}

def train_agent_on_window(train_df, train_features, total_timesteps, params, pretrained_policy=None, seed=42, reward_scheme='sortino'):
    env = TradingEnvironment(df=train_df, surprise_scores=train_features, reward_scheme=reward_scheme)
    agent = PPOAgent(env, lr=params.get('lr', 3e-4), entropy_coef=params.get('entropy_coef', 0.01))
    if pretrained_policy:
        print("    -> Loading pre-trained specialist weights...")
        agent.policy.load_state_dict(pretrained_policy.state_dict())
    state, _ = env.reset(seed=seed)
    for timestep in tqdm(range(total_timesteps), desc="    Training"):
        old_state, action, log_prob, value = agent.select_action(state)
        next_state, reward, done, _, _ = env.step(action)
        agent.buffer.states.append(old_state)
        agent.buffer.actions.append(action)
        agent.buffer.logprobs.append(log_prob)
        agent.buffer.rewards.append(reward)
        agent.buffer.is_terminals.append(done)
        agent.buffer.values.append(value)
        state = next_state
        if done:
            agent.learn()
            state, _ = env.reset(seed=seed + timestep + 1)
    return agent.policy

def test_agent_on_window(policy_net, test_df, test_features, initial_balance, regime_labels, seed=42, reward_scheme='sortino'):
    test_env = TradingEnvironment(df=test_df, surprise_scores=test_features, initial_balance=initial_balance, reward_scheme=reward_scheme)
    state, _ = test_env.reset(seed=seed)
    log = [{'Timestamp': test_df.index[0] - pd.Timedelta(hours=1), 'PortfolioValue': initial_balance, 'AssetPrice': test_df.iloc[0]['Open'], 'Regime': regime_labels[0]}]
    done, cs = False, 0
    with tqdm(total=len(test_df), desc="    Testing ") as pbar:
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
            current_label = regime_labels[cs] if cs < len(regime_labels) else regime_labels[-1]
            log.append({'Timestamp': test_df.index[cs], 'PortfolioValue': info['portfolio_value'], 'AssetPrice': test_df.iloc[cs]['Close'], 'Regime': current_label})
            cs += 1; pbar.update(1)
    return pd.DataFrame(log)

def run_walk_forward_analysis(full_df, all_features, total_training_steps, params, save_plot_filename=None, seed=42):
    train_ws, test_ws, hist_lb = 63 * 7, 21 * 7, 4
    init_b, curr_b = 100000, 100000
    all_res = []
    num_w = (len(all_features) - train_ws) // test_ws
    spec_pols = {}
    log_filename = 'regime_log_v21.txt'
    with open(log_filename, 'w') as f: f.write("--- v21 Ground Truth Regime Log ---\n")

    print(f"\n--- Starting v21 Backtest ({num_w} windows) ---")
    for i in range(hist_lb, num_w):
        window_num = i + 1 - hist_lb
        print(f"\n--- Window {window_num}/{num_w - hist_lb} ---")
        print("  Detecting regimes with PCA + BGMM...")
        hist_si, hist_ei = (i - hist_lb) * test_ws, i * test_ws + train_ws
        historical_vectors = all_features[hist_si:hist_ei]
        scaler, pca, bgmm, n_reg = detect_regimes(historical_vectors, max_regimes=8, seed=seed)

        if bgmm is None:
            with open(log_filename, 'a') as f: f.write(f"Window {window_num}: SKIPPED\n"); continue
        
        reg_map, _ = map_regimes_to_labels(scaler, pca, bgmm, n_reg)
        print(f"  Discovered {n_reg} regimes. Labels: {sorted(list(reg_map.values()))}")

        train_si, train_ei = i * test_ws, i * test_ws + train_ws
        train_df, train_feat = full_df.iloc[train_si:train_ei], all_features[train_si:train_ei]

        current_vectors_scaled = scaler.transform(train_feat)
        current_vectors_reduced = pca.transform(current_vectors_scaled)
        regime_predictions = bgmm.predict(current_vectors_reduced)
        
        active_regime_keys = list(reg_map.keys())
        filtered_predictions = [p for p in regime_predictions if p in active_regime_keys]
        
        if filtered_predictions:
            counts = np.bincount(filtered_predictions)
            curr_reg_int = np.argmax(counts)
        else:
            curr_reg_int = active_regime_keys[0]

        mapped_label = reg_map.get(curr_reg_int, "UNKNOWN")

        rew_sch = 'sharpe' if mapped_label == 'BULL' else 'sortino'
        print(f"  Current Dominant Regime: '{mapped_label}'. Using '{rew_sch}' reward.")
        
        if 'NEUTRAL' in mapped_label:
            current_params = params['NEUTRAL']
        else:
            current_params = params.get(mapped_label, list(params.values())[0])
        
        active_specialist_policy = spec_pols.get(mapped_label)
        if active_specialist_policy is None: print(f"    -> Training NEW '{mapped_label}' specialist.")
        else: print(f"    -> Fine-tuning EXISTING '{mapped_label}' specialist.")

        trained_pol = train_agent_on_window(train_df, train_feat, total_training_steps, current_params, active_specialist_policy, seed + i, rew_sch)
        spec_pols[mapped_label] = trained_pol

        test_si, test_ei = train_ei, train_ei + test_ws
        if test_ei > len(all_features): break
        test_df, test_feat = full_df.iloc[test_si:test_ei], all_features[test_si:test_ei]

        test_vectors_scaled = scaler.transform(test_feat)
        test_vectors_reduced = pca.transform(test_vectors_scaled)
        test_regime_ints = bgmm.predict(test_vectors_reduced)
        test_regime_labels = [reg_map.get(p, "UNKNOWN") for p in test_regime_ints]

        month_res = test_agent_on_window(trained_pol, test_df, test_feat, curr_b, test_regime_labels, seed + i, rew_sch)
        all_res.append(month_res.iloc[1:])
        
        if not month_res.empty:
            curr_b = month_res['PortfolioValue'].iloc[-1]
            print(f"  ==> End of Window: Portfolio Value = ${curr_b:,.2f}")

        with open(log_filename, 'a') as f: f.write(f"Window {window_num}: Discovered {n_reg} regimes. Dominant: {mapped_label}.\n")

    print("\n--- v21 Walk-Forward Backtest Complete ---")
    if not all_res: return pd.DataFrame()
    final_res = pd.concat(all_res)
    start_row = pd.DataFrame([{'Timestamp': final_res['Timestamp'].iloc[0] - pd.Timedelta(hours=1), 'PortfolioValue': init_b, 'AssetPrice': final_res['AssetPrice'].iloc[0], 'Regime': final_res['Regime'].iloc[0]}])
    res_df = pd.concat([start_row, final_res]).set_index('Timestamp')

    if save_plot_filename and not res_df.empty:
        print(f"\nSaving plot to '{save_plot_filename}'...")
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(16, 8))
        norm_p = (res_df['PortfolioValue'] / res_df['PortfolioValue'].iloc[0]) * 100
        norm_a = (res_df['AssetPrice'] / res_df['AssetPrice'].iloc[0]) * 100
        ax.plot(norm_p, label='v21 Agent Equity Curve', color='deepskyblue', linewidth=2)
        ax.plot(norm_a, label='QQQ Asset Price (Buy & Hold)', color='gray', linestyle='--', linewidth=1.5)

        unique_regimes = sorted(list(res_df['Regime'].unique()))
        colors = plt.cm.get_cmap('viridis', len(unique_regimes))
        label_to_color = {label: colors(i) for i, label in enumerate(unique_regimes)}
        
        leg_p = [mpatches.Patch(color=c, label=f"Regime: {l}") for l, c in label_to_color.items()]
        start_date, current_regime_label = res_df.index[0], res_df['Regime'].iloc[0]
        for i in range(1, len(res_df)):
            if res_df['Regime'].iloc[i] != current_regime_label:
                end_date = res_df.index[i]
                ax.axvspan(start_date, end_date, color=label_to_color.get(current_regime_label, 'white'), alpha=0.3)
                start_date, current_regime_label = end_date, res_df['Regime'].iloc[i]
        ax.axvspan(start_date, res_df.index[-1], color=label_to_color.get(current_regime_label, 'white'), alpha=0.3)

        main_leg = ax.legend(loc='upper left'); ax.add_artist(main_leg)
        ax.legend(handles=leg_p, loc='lower right', title='Market Regimes (Discovered)')
        ax.set_title("v21 'Self-Discovering Regimes' Performance", fontsize=16)
        ax.set_xlabel("Date"); ax.set_ylabel("Normalized Value (Start = 100)")
        fig.tight_layout(); plt.savefig(save_plot_filename); plt.close()
        print("Plot saved.")
    return res_df


if __name__ == '__main__':
    SEED = 42
    set_seeds(SEED)
    all_data = fetch_and_prepare_data(ticker="QQQ", period="729d", interval="1h")
    if all_data:
        features, df = calculate_walk_forward_features(all_data)
        
        # --- DEFINITIVE PARAMETERS from the user's final console output ---
        v21_baseline_params = {
            'BULL': {'lr': 0.000052, 'entropy_coef': 0.0582},
            'BEAR': {'lr': 0.000085, 'entropy_coef': 0.0999},
            'NEUTRAL': {'lr': 0.000011, 'entropy_coef': 0.0164} # Mapped from v20 SIDEWAYS
        }
        
        results = run_walk_forward_analysis(
            df, features, total_training_steps=50000, params=v21_baseline_params,
            save_plot_filename='equity_curve_v21_baseline.png', seed=SEED
        )
        if not results.empty:
            print(f"\n--- FINAL v21 BASELINE RESULT ---\nFinal Portfolio Value: ${results['PortfolioValue'].iloc[-1]:,.2f}")