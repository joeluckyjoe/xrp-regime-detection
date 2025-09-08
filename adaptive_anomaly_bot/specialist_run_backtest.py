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

def calculate_walk_forward_features(dataframes):  
    print("Calculating v19 walk-forward multi-scale features (108 per step)...")  
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

def train_agent_on_window(train_df, train_features, total_timesteps, params, pretrained_policy=None, seed=42, reward_scheme='sortino'):  
    env = TradingEnvironment(df=train_df, surprise_scores=train_features, reward_scheme=reward_scheme)  
    agent = PPOAgent(env, lr=params.get('lr', 3e-4), entropy_coef=params.get('entropy_coef', 0.01))  
    if pretrained_policy:  
        print("    --> Loading pre-trained specialist weights..."); agent.policy.load_state_dict(pretrained_policy.state_dict())  
    state, _ = env.reset(seed=seed)  
    for timestep in tqdm(range(total_timesteps), desc="    Training"):  
        old_state, action, log_prob, value = agent.select_action(state); next_state, reward, done, _, _ = env.step(action)  
        agent.buffer.states.append(old_state);agent.buffer.actions.append(action);agent.buffer.logprobs.append(log_prob)  
        agent.buffer.rewards.append(reward);agent.buffer.is_terminals.append(done);agent.buffer.values.append(value)  
        state=next_state  
        if done: agent.learn(); state, _ = env.reset(seed=seed + timestep + 1)  
    return agent.policy

def test_agent_on_window(policy_net, test_df, test_features, initial_balance, mapped_label, seed=42, reward_scheme='sortino'):  
    test_env = TradingEnvironment(df=test_df, surprise_scores=test_features, initial_balance=initial_balance, reward_scheme=reward_scheme)  
    state, _ = test_env.reset(seed=seed)  
    log = [{'Timestamp': test_df.index[0] - pd.Timedelta(hours=1), 'PortfolioValue': initial_balance, 'AssetPrice': test_df.iloc[0]['Open'], 'Regime': mapped_label}]  
    done, cs = False, 0  
    with tqdm(total=len(test_df), desc="    Testing ") as pbar:  
        while not done:  
            st = torch.FloatTensor(state).unsqueeze(0)  
            with torch.no_grad():  
                ap,mn,_,_ = policy_net(st); da=torch.argmax(ap).item(); cat=mn  
                low,high=torch.tensor(test_env.action_space[1].low),torch.tensor(test_env.action_space[1].high)  
                cas=low+(cat+1.0)*0.5*(high-low); act=(da, cas.squeeze(0).numpy())  
            state, _, done, _, info = test_env.step(act)  
            log.append({'Timestamp': test_df.index[cs], 'PortfolioValue': info['portfolio_value'], 'AssetPrice': test_df.iloc[cs]['Close'], 'Regime': mapped_label})  
            cs += 1; pbar.update(1)  
    return pd.DataFrame(log)

def run_walk_forward_analysis(full_df, all_features, total_training_steps, params, save_plot_filename=None, seed=42):  
    train_ws, test_ws, hist_lb, n_reg = 63*7, 21*7, 4, 3  
    init_b, curr_b = 100000, 100000; all_res = []  
    num_w = (len(all_features) - train_ws) // test_ws  
    spec_pols = {'BULL': None, 'BEAR': None, 'SIDEWAYS': None}  
      
    log_filename = 'regime_log.txt'  
    with open(log_filename, 'w') as f:  
        f.write("--- Ground Truth Regime Log ---\n")

    print(f"\n--- Starting v21 'Team of Specialists' Backtest ({num_w} windows) ---")  
    for i in range(hist_lb, num_w):  
        window_num = i + 1 - hist_lb  
        print(f"\n--- Window {window_num}/{num_w-hist_lb} ---")  
        print("  Detecting regimes..."); hist_w=[]  
        for j in range(i-hist_lb, i): hist_w.append(full_df.iloc[j*test_ws : j*test_ws+train_ws])  
        scaler, kmeans = detect_regimes(hist_w, n_regimes=n_reg, seed=seed)  
        if kmeans is None:   
            print("  Skipping window due to insufficient data for clustering.")  
            with open(log_filename, 'a') as f: f.write(f"Window {window_num}: SKIPPED (Insufficient Data)\n")  
            continue  
        reg_map = map_regimes_to_labels(kmeans, scaler)  
        train_si, train_ei = i*test_ws, i*test_ws+train_ws  
        train_df, train_feat = full_df.iloc[train_si:train_ei], all_features[train_si:train_ei]  
        curr_fp = create_window_fingerprint(train_df); scaled_fp = scaler.transform(curr_fp.reshape(1, -1))  
        curr_reg_int = kmeans.predict(scaled_fp)[0]; mapped_label = reg_map[curr_reg_int]  
        # v21: Use 'calmar' for BEAR and SIDEWAYS regimes instead of 'sortino'
        rew_sch = 'sharpe' if mapped_label == 'BULL' else 'calmar'  
        print(f"  Regime Identified: '{mapped_label}'. Using '{rew_sch}' reward.")  
        act_spec = spec_pols[mapped_label]  
        if act_spec is None: print(f"    --> Training NEW '{mapped_label}' specialist.")  
        else: print(f"    --> Fine-tuning EXISTING '{mapped_label}' specialist.")  
        current_params = params.get(mapped_label, params)  
        trained_pol = train_agent_on_window(train_df, train_feat, total_training_steps, current_params, act_spec, seed+i, rew_sch)  
        spec_pols[mapped_label] = trained_pol  
          
        test_si, test_ei = train_ei, train_ei+test_ws  
        if test_ei > len(all_features): break  
        test_df, test_feat = full_df.iloc[test_si:test_ei], all_features[test_si:test_ei]  
          
        log_price = np.log(test_df['Close'].replace(0, 1e-9))  
        x = np.arange(len(log_price)); A = np.vstack([x, np.ones(len(x))]).T  
        window_momentum, _ = np.linalg.lstsq(A, log_price, rcond=None)[0]  
          
        month_res = test_agent_on_window(trained_pol, test_df, test_feat, curr_b, mapped_label, seed+i, rew_sch)  
        all_res.append(month_res.iloc[1:])  
        if not month_res.empty:  
            curr_b = month_res['PortfolioValue'].iloc[-1]; print(f"  ==> End of Window: Portfolio Value = ${curr_b:,.2f}")  
          
        with open(log_filename, 'a') as f:  
            f.write(f"Window {window_num}: {mapped_label} (Actual Momentum: {window_momentum:.4f})\n")

    print("\n--- Walk-Forward Backtest Complete ---")  
    if not all_res: return pd.DataFrame()  
    final_res = pd.concat(all_res)  
    start_row = pd.DataFrame([{'Timestamp': final_res['Timestamp'].iloc[0]-pd.Timedelta(hours=1), 'PortfolioValue': init_b, 'AssetPrice': final_res['AssetPrice'].iloc[0], 'Regime': final_res['Regime'].iloc[0]}])  
    res_df = pd.concat([start_row, final_res]).set_index('Timestamp')  
    if save_plot_filename and not res_df.empty:  
        print(f"\nSaving plot to '{save_plot_filename}'...")  
        plt.style.use('seaborn-v0_8-whitegrid'); fig,ax = plt.subplots(figsize=(16,8))  
        norm_p=(res_df['PortfolioValue']/res_df['PortfolioValue'].iloc[0])*100  
        norm_a=(res_df['AssetPrice']/res_df['AssetPrice'].iloc[0])*100  
        ax.plot(norm_p, label='v21 Calmar Agent Equity Curve', color='deepskyblue', linewidth=2)  
        ax.plot(norm_a, label='QQQ Asset Price (Buy & Hold)', color='gray', linestyle='--', linewidth=1.5)  
        label_to_color = {'BULL': '#E6F3FF', 'BEAR': '#FFE6E6', 'SIDEWAYS': '#F0F0F0'}  
        leg_p = [mpatches.Patch(color=c, label=f"Regime: {l.capitalize()}", alpha=0.6) for l, c in label_to_color.items()]  
        start_date = res_df.index[0]; current_regime_label = res_df['Regime'].iloc[0]  
        for i in range(1, len(res_df)):  
            if res_df['Regime'].iloc[i] != current_regime_label:  
                end_date = res_df.index[i]  
                ax.axvspan(start_date, end_date, color=label_to_color.get(current_regime_label, 'white'), alpha=0.6)  
                start_date = end_date; current_regime_label = res_df['Regime'].iloc[i]  
        ax.axvspan(start_date, res_df.index[-1], color=label_to_color.get(current_regime_label, 'white'), alpha=0.6)  
        main_leg=ax.legend(loc='upper left'); ax.add_artist(main_leg); ax.legend(handles=leg_p, loc='lower right', title='Market Regimes')  
        ax.set_title("v21 'Team of Specialists' Performance (Calmar Reward)", fontsize=16)  
        ax.set_xlabel("Date", fontsize=12); ax.set_ylabel("Normalized Value (Start = 100)", fontsize=12); fig.tight_layout(); plt.savefig(save_plot_filename); plt.close()  
        print("Plot saved.")  
    return res_df

if __name__ == '__main__':  
    SEED = 42; set_seeds(SEED)  
    all_data = fetch_and_prepare_data(ticker="QQQ", period="729d", interval="1h")  
    if all_data:  
        features, df = calculate_walk_forward_features(all_data)  
        # --- Using the final CHAMPION parameters from the v20 optimization ---  
        params = {  
            'BULL':     {'lr': 0.000052, 'entropy_coef': 0.0582},  
            'BEAR':     {'lr': 0.000085, 'entropy_coef': 0.0999},  
            'SIDEWAYS': {'lr': 0.000011, 'entropy_coef': 0.0164}  
        }  
        results = run_walk_forward_analysis(df, features, 50000, params, 'equity_curve_v21_calmar_test.png', SEED)  
        if not results.empty:  
            print(f"\n--- FINAL RESULT ---\nFinal Portfolio Value: ${results['PortfolioValue'].iloc[-1]:,.2f}")