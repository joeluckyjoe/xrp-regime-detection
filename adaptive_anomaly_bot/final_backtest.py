import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
import gymnasium as gym
from tqdm import tqdm
import os
import pickle
import yfinance as yf
import itertools
import matplotlib.pyplot as plt
import warnings

# --- Global Configuration ---
TICKER = "QQQ"
PERIOD = "729d"
INTERVAL = "1h"
OUTPUT_DIR = "final_backtest_results"
# Walk-Forward Configuration
WINDOW_SIZE = 3535 
STEP_SIZE = 168 
# Layer 1 Config
L1_INDICATORS = ['returns','Volume','rsi','atr', 'MACD_12_26_9','MACDh_12_26_9','MACDs_12_26_9','ADX_14']
L1_TIMEFRAMES = ['1h', '1d', '1w']
L1_WINDOW_SIZES = {'1h': 882, '1d': 126, '1w': 26}
# Layer 2 Config
VOLATILITY_WINDOW = 24
N_STATES = 2
# Layer 3 Config
TRAINING_EPISODES = 50
UPDATE_TIMESTEP = 2000

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# --- Helper Functions and Classes ---
def calculate_rsi(close, length=14):
    delta = close.diff(); gain = (delta.where(delta > 0, 0)).rolling(window=length).mean(); loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean(); rs = gain / loss
    return 100 - (100 / (1 + rs))
def calculate_atr(high, low, close, length=14):
    tr1 = pd.DataFrame(high - low); tr2 = pd.DataFrame(abs(high - close.shift(1))); tr3 = pd.DataFrame(abs(low - close.shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1); atr = tr.ewm(alpha=1/length, adjust=False).mean()
    return atr / close * 100
def calculate_macd(close, fast=12, slow=26, signal=9):
    exp1 = close.ewm(span=fast, adjust=False).mean(); exp2 = close.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2; signal_line = macd.ewm(span=signal, adjust=False).mean(); histogram = macd - signal_line
    return macd, signal_line, histogram
def calculate_adx(high, low, close, length=14):
    plus_dm = high.diff(); minus_dm = low.diff(); plus_dm[plus_dm < 0] = 0; minus_dm[minus_dm > 0] = 0
    tr1 = pd.DataFrame(high - low); tr2 = pd.DataFrame(abs(high - close.shift(1))); tr3 = pd.DataFrame(abs(low - close.shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1); atr = tr.ewm(alpha=1/length, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/length, adjust=False).mean() / atr)
    minus_di = 100 * (abs(minus_dm.ewm(alpha=1/length, adjust=False).mean()) / atr)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di)); adx = dx.ewm(alpha=1/length, adjust=False).mean()
    return adx
def calculate_sharpe_ratio(returns, periods_per_year=252*7):
    if returns.std() == 0: return 0.0
    return np.sqrt(periods_per_year) * returns.mean() / returns.std()
def calculate_max_drawdown(equity_curve):
    running_max = equity_curve.cummax(); drawdown = (equity_curve - running_max) / running_max
    return drawdown.min()
class BayesianAnomalyDetector:
    def __init__(self, distribution_type='t'): self.dist_type = distribution_type; self.alpha_0, self.beta_0, self.mu_0, self.nu_0 = 1.0, 1.0, 0.0, 1.0; self.reset_posterior()
    def reset_posterior(self): self.alpha_n, self.beta_n, self.mu_n, self.nu_n = self.alpha_0, self.beta_0, self.mu_0, self.nu_0
    def fit(self, data):
        if self.dist_type == 't': self._fit_t(data)
    def compute_surprise(self, x):
        from scipy.stats import t
        df,loc,scale=2*self.alpha_n,self.mu_n,np.sqrt(self.beta_n/(self.alpha_n*self.nu_n));
        if scale <= 0 or not np.isfinite(scale): return 0.5
        return 1.0 - (2 * min(t.cdf(x,df=df,loc=loc,scale=scale), t.sf(x,df=df,loc=loc,scale=scale)))
    def get_distribution_params(self):
        scale = np.sqrt(self.beta_n / (self.alpha_n * self.nu_n)); return self.mu_n, scale
    def _fit_t(self, data):
        n=len(data);
        if n==0: return
        mean_data=data.mean(); sum_sq_diff=((data-mean_data)**2).sum()
        self.alpha_n=self.alpha_0+n/2; self.beta_n=self.beta_0+0.5*sum_sq_diff+(n*self.nu_0)/(self.nu_0+n)*0.5*(mean_data-self.mu_0)**2
        self.mu_n=(self.nu_0*self.mu_0+n*mean_data)/(self.nu_0+n); self.nu_n=self.nu_0+n
class RollingNormalizer:
    def __init__(self, window_size=252): self.window_size, self.data, self.min, self.max = window_size, deque(maxlen=window_size), None, None
    def update(self, value): self.data.append(value); self.min, self.max = min(self.data), max(self.data)
    def normalize(self, value): return 0.5 if self.max is None or self.min is None or self.max == self.min else (value - self.min) / (self.max - self.min)

def get_features_and_ohlc(full_data_df):
    agg_logic = {'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}
    df_1d = full_data_df.resample('D').agg(agg_logic).dropna()
    df_1w = full_data_df.resample('W-MON').agg(agg_logic).dropna()
    
    dataframes = {'1h': full_data_df, '1d': df_1d, '1w': df_1w}; processed_dfs = {}
    for timeframe, df in dataframes.items():
        df_processed = df.copy(); df_processed['returns'] = (df_processed['Close'] - df_processed['Open']) / df_processed['Open']
        df_processed['rsi'] = calculate_rsi(df_processed['Close']) / 100.0
        df_processed['atr'] = calculate_atr(df_processed['High'], df_processed['Low'], df_processed['Close'])
        macd, macds, macdh = calculate_macd(df_processed['Close']); df_processed['MACD_12_26_9'] = macd; df_processed['MACDs_12_26_9'] = macds; df_processed['MACDh_12_26_9'] = macdh
        df_processed['ADX_14'] = calculate_adx(df_processed['High'], df_processed['Low'], df_processed['Close'])
        df_processed.dropna(inplace=True)
        processed_dfs[timeframe] = df_processed

    # --- FIX: Move the length check to after the processing and dropna() ---
    if len(processed_dfs['1w']) < L1_WINDOW_SIZES['1w']:
        return None, None
    
    detectors,normalizers,data_windows={},{},{}
    for tf,ind in itertools.product(L1_TIMEFRAMES,L1_INDICATORS):
        key=f'{tf}_{ind}';win_size=L1_WINDOW_SIZES[tf]; detectors[key]=BayesianAnomalyDetector(distribution_type='t'); normalizers[f'{key}_raw']=RollingNormalizer(win_size); normalizers[f'{key}_mean']=RollingNormalizer(win_size); normalizers[f'{key}_scale']=RollingNormalizer(win_size); data_windows[key]=list(processed_dfs[tf][ind].iloc[:win_size])
    for key,window in data_windows.items():
        detectors[key].fit(pd.Series(window)); [normalizers[f'{key}_raw'].update(val) for val in window]
    
    start_ts=processed_dfs['1w'].index[L1_WINDOW_SIZES['1w']]; start_index=processed_dfs['1h'].index.searchsorted(start_ts)
    asset_features=[]
    for i in range(start_index,len(processed_dfs['1h'])):
        step_features=[]; current_ts=processed_dfs['1h'].index[i]
        for tf in L1_TIMEFRAMES:
            tf_loc=processed_dfs[tf].index.searchsorted(current_ts,side='right')-1
            if tf_loc < 0: continue
            current_observation=processed_dfs[tf].iloc[tf_loc]
            for ind in L1_INDICATORS:
                key=f'{tf}_{ind}';current_value=current_observation[ind]
                if len(data_windows[key]) <= tf_loc:
                    data_windows.setdefault(key, []).append(current_value)
                    if len(data_windows[key])>L1_WINDOW_SIZES[tf]: data_windows[key].pop(0)
                    detectors[key].fit(pd.Series(data_windows[key]))
                surprise=detectors[key].compute_surprise(current_value); mean,scale=detectors[key].get_distribution_params()
                normalizers[f'{key}_raw'].update(current_value); norm_raw=normalizers[f'{key}_raw'].normalize(current_value); normalizers[f'{key}_mean'].update(mean); norm_mean=normalizers[f'{key}_mean'].normalize(mean); normalizers[f'{key}_scale'].update(scale); norm_scale=normalizers[f'{key}_scale'].normalize(scale)
                step_features.extend([surprise,norm_raw,norm_mean,norm_scale])
        if len(step_features) == 96: asset_features.append(step_features)
    
    feature_array = np.array(asset_features, dtype=np.float32)
    aligned_df = processed_dfs['1h'].iloc[start_index:start_index+len(asset_features)]
    return feature_array, aligned_df

class VolatilityBreakoutEnv(gym.Env):
    def __init__(self, features, market_data, initial_balance=100000, trade_fee=0.001):
        super().__init__(); self.features = features; self.df = market_data.reset_index(drop=True); self.initial_balance = initial_balance; self.trade_fee = trade_fee; self.action_space = gym.spaces.Discrete(3); self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(features.shape[1],), dtype=np.float32)
    def reset(self, seed=None):
        super().reset(seed=seed); self._current_step = 1; self.balance = self.initial_balance; self.shares_held = 0; self.entry_price = 0; self.position = 0; self.entry_portfolio_value = 0; self.done = False
        if len(self.features) <= 1: self.done=True; return self.features[0] if len(self.features) > 0 else np.zeros(self.observation_space.shape), {}
        return self.features[self._current_step], {}
    def step(self, action):
        current_price = self.df.iloc[self._current_step]['Close']; is_breakout = (self.df.iloc[self._current_step - 1]['regime'] == 0 and self.df.iloc[self._current_step]['regime'] == 1)
        if self.position == 0 and is_breakout:
            if action == 1 or action == 2:
                self.entry_portfolio_value = self._get_portfolio_value(); self.shares_held = self.balance * (1 - self.trade_fee) / current_price; self.balance = 0; self.position = 1 if action == 1 else -1; self.entry_price = current_price
        elif self.position != 0:
            back_to_compression = self.df.iloc[self._current_step]['regime'] == 0; agent_exits = action == 0
            if back_to_compression or agent_exits:
                if self.position == 1: self.balance = self.shares_held * current_price * (1 - self.trade_fee)
                elif self.position == -1: profit_loss = (self.entry_price - current_price) * self.shares_held; self.balance = self.entry_portfolio_value + profit_loss; self.balance *= (1 - self.trade_fee)
                self.shares_held = 0; self.position = 0; self.entry_price = 0
        self._current_step += 1
        if self._current_step >= len(self.features) - 1: self.done = True
        return self.features[self._current_step], 0, self.done, False, {}
    def _get_portfolio_value(self):
        if self._current_step >= len(self.df): return self.balance
        current_price = self.df.iloc[self._current_step]['Close']
        if self.position == 1: return self.shares_held * current_price
        elif self.position == -1: profit_loss = (self.entry_price - current_price) * self.shares_held; return self.entry_portfolio_value + profit_loss
        else: return self.balance
class ActorCritic(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(ActorCritic, self).__init__(); self.actor=nn.Sequential(nn.Linear(n_observations,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,n_actions),nn.Softmax(dim=-1)); self.critic=nn.Sequential(nn.Linear(n_observations,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,1))
    def forward(self, x): return self.actor(x), self.critic(x)
class RolloutBuffer:
    def __init__(self): self.actions, self.states, self.logprobs, self.rewards, self.is_terminals, self.values = [], [], [], [], [], []
    def clear(self): del self.actions[:]; del self.states[:]; del self.logprobs[:]; del self.rewards[:]; del self.is_terminals[:]; del self.values[:]
class PPOAgent:
    def __init__(self, n_obs, n_actions, lr=3e-4, gamma=0.99, K_epochs=4, eps_clip=0.2):
        self.gamma=gamma; self.eps_clip=eps_clip; self.K_epochs=K_epochs; self.policy=ActorCritic(n_obs,n_actions); self.optimizer=optim.Adam(self.policy.parameters(),lr=lr); self.policy_old=ActorCritic(n_obs,n_actions); self.policy_old.load_state_dict(self.policy.state_dict()); self.MseLoss=nn.MSELoss(); self.buffer=RolloutBuffer()
    def select_action(self, state, is_eval=False):
        with torch.no_grad():
            state_tensor=torch.FloatTensor(state); action_probs,state_val=self.policy_old(state_tensor); dist=Categorical(action_probs)
            if is_eval: action = torch.argmax(action_probs)
            else: action=dist.sample()
            return action.item(),dist.log_prob(action).item(),state_val.item()
    def learn(self):
        rewards = []; discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal: discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward); rewards.insert(0, discounted_reward)
        rewards = torch.tensor(rewards, dtype=torch.float32); rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        old_states = torch.squeeze(torch.tensor(np.array(self.buffer.states), dtype=torch.float32)); old_actions = torch.squeeze(torch.tensor(self.buffer.actions, dtype=torch.long)); old_logprobs = torch.squeeze(torch.tensor(self.buffer.logprobs, dtype=torch.float32))
        for _ in range(self.K_epochs):
            action_probs, state_values = self.policy(old_states); dist = Categorical(action_probs); logprobs = dist.log_prob(old_actions); dist_entropy = dist.entropy(); state_values = torch.squeeze(state_values)
            advantages = rewards - state_values.detach(); ratios = torch.exp(logprobs - old_logprobs)
            surr1 = ratios * advantages; surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            self.optimizer.zero_grad(); loss.mean().backward(); self.optimizer.step()
        self.policy_old.load_state_dict(self.policy.state_dict()); self.buffer.clear()

# --- Main Backtesting Loop ---
if __name__ == "__main__":
    from hmmlearn.hmm import GaussianHMM
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Fetching full dataset...")
    full_data = yf.Ticker(TICKER).history(period=PERIOD, interval=INTERVAL, auto_adjust=False)
    
    full_equity_curve = pd.Series(dtype=np.float64)
    start_index = 0
    
    num_windows = (len(full_data) - WINDOW_SIZE) // STEP_SIZE
    if num_windows <= 0:
        raise ValueError("Not enough data for a single walk-forward window. Please check PERIOD, WINDOW_SIZE, and STEP_SIZE.")
    pbar = tqdm(total=num_windows)
    
    while start_index + WINDOW_SIZE + STEP_SIZE <= len(full_data):
        train_df = full_data.iloc[start_index : start_index + WINDOW_SIZE]
        test_df = full_data.iloc[start_index + WINDOW_SIZE : start_index + WINDOW_SIZE + STEP_SIZE]
        pbar.set_description(f"Processing window starting at {train_df.index[0].date()}")
        
        train_features, train_aligned_df = get_features_and_ohlc(train_df)
        if train_features is None:
            start_index += STEP_SIZE; pbar.update(1); continue
        
        train_returns = train_aligned_df['Close'].pct_change()
        train_vol = train_returns.rolling(window=VOLATILITY_WINDOW).std().dropna().values.reshape(-1, 1)
        hmm_model = GaussianHMM(n_components=N_STATES, covariance_type="full", n_iter=100, random_state=42)
        hmm_model.fit(train_vol)
        
        train_vol_series = train_returns.rolling(window=VOLATILITY_WINDOW).std().dropna()
        train_regimes_raw = hmm_model.predict(train_vol_series.values.reshape(-1, 1))
        low_vol_state = np.argmin(hmm_model.means_)
        train_regimes_ordered = np.where(train_regimes_raw == low_vol_state, 0, 1)
        regime_series = pd.Series(train_regimes_ordered, index=train_vol_series.index, name="regime")
        train_env_df = train_aligned_df.join(regime_series)
        train_env_df.dropna(inplace=True)
        final_train_features = train_features[-len(train_env_df):]
        
        rl_env = VolatilityBreakoutEnv(final_train_features, train_env_df)
        agent = PPOAgent(rl_env.observation_space.shape[0], rl_env.action_space.n)
        time_step=0
        for _ in range(TRAINING_EPISODES):
            state, _ = rl_env.reset()
            if rl_env.done: continue
            done = False
            while not done:
                time_step += 1; action, log_prob, state_val = agent.select_action(state); next_state, reward, done, _, _ = rl_env.step(action)
                agent.buffer.states.append(state); agent.buffer.actions.append(action); agent.buffer.logprobs.append(log_prob); agent.buffer.rewards.append(reward); agent.buffer.is_terminals.append(done); agent.buffer.values.append(state_val)
                state = next_state
                if time_step % UPDATE_TIMESTEP == 0:
                    agent.learn(); time_step=0
        
        combined_window_df = pd.concat([train_df.iloc[-L1_WINDOW_SIZES['1h']:], test_df])
        test_features, test_aligned_df = get_features_and_ohlc(combined_window_df)
        if test_features is None:
            start_index += STEP_SIZE; pbar.update(1); continue
        test_features = test_features[-len(test_df):]; test_aligned_df = test_aligned_df.iloc[-len(test_df):]
        
        test_returns = test_aligned_df['Close'].pct_change()
        test_vol_series = test_returns.rolling(window=VOLATILITY_WINDOW).std().dropna()
        if test_vol_series.empty:
            start_index += STEP_SIZE; pbar.update(1); continue
            
        test_regimes_raw = hmm_model.predict(test_vol_series.values.reshape(-1, 1))
        test_regimes_ordered = np.where(test_regimes_raw == low_vol_state, 0, 1)
        regime_series_test = pd.Series(test_regimes_ordered, index=test_vol_series.index, name="regime")
        test_env_df = test_aligned_df.join(regime_series_test)
        test_env_df.dropna(inplace=True)
        final_test_features = test_features[-len(test_env_df):]
        
        if len(final_test_features) == 0:
            start_index += STEP_SIZE; pbar.update(1); continue
        
        eval_env = VolatilityBreakoutEnv(final_test_features, test_env_df, initial_balance=full_equity_curve.iloc[-1] if not full_equity_curve.empty else 100000)
        state, _ = eval_env.reset()
        if eval_env.done:
            start_index += STEP_SIZE; pbar.update(1); continue
            
        done = False
        equity_chunk = [eval_env.initial_balance]
        while not done:
            action, _, _ = agent.select_action(state, is_eval=True)
            state, _, done, _, _ = eval_env.step(action)
            equity_chunk.append(eval_env._get_portfolio_value())
            
        chunk_equity_series = pd.Series(equity_chunk, index=test_env_df.index[:len(equity_chunk)])
        if full_equity_curve.empty:
            full_equity_curve = chunk_equity_series
        else:
            full_equity_curve = pd.concat([full_equity_curve.iloc[:-1], chunk_equity_series])
            
        start_index += STEP_SIZE
        pbar.update(1)
    pbar.close()

    print("Backtest complete. Calculating final performance...")
    
    if full_equity_curve.empty:
        print("No trades were executed during the backtest period. Cannot generate performance report.")
    else:
        agent_returns = full_equity_curve.pct_change().dropna()
        bh_series = full_data['Close'].loc[full_equity_curve.index]
        buy_hold_equity = full_equity_curve.iloc[0] * (bh_series / bh_series.iloc[0])
        buy_hold_returns = buy_hold_equity.pct_change().dropna()

        agent_total_return = (full_equity_curve.iloc[-1] / full_equity_curve.iloc[0] - 1) * 100
        bh_total_return = (buy_hold_equity.iloc[-1] / buy_hold_equity.iloc[0] - 1) * 100
        agent_sharpe = calculate_sharpe_ratio(agent_returns)
        bh_sharpe = calculate_sharpe_ratio(buy_hold_returns)
        agent_mdd = calculate_max_drawdown(full_equity_curve) * 100
        bh_mdd = calculate_max_drawdown(buy_hold_equity) * 100

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.plot(full_equity_curve.index, full_equity_curve / full_equity_curve.iloc[0], label="Adaptive Agent", color='royalblue', lw=2)
        ax.plot(buy_hold_equity.index, buy_hold_equity / buy_hold_equity.iloc[0], label="Buy & Hold Baseline", color='gray', linestyle='--')
        ax.set_title(f"Rolling Walk-Forward Backtest for {TICKER}", fontsize=16)
        ax.set_ylabel("Normalized Portfolio Value", fontsize=12)
        ax.legend(loc='upper left', fontsize=12)
        stats_text = (f"Agent Performance:\n------------------\nTotal Return: {agent_total_return:.2f}%\nSharpe Ratio: {agent_sharpe:.2f}\nMax Drawdown: {agent_mdd:.2f}%\n\nBuy & Hold Performance:\n----------------------\nTotal Return: {bh_total_return:.2f}%\nSharpe Ratio: {bh_sharpe:.2f}\nMax Drawdown: {bh_mdd:.2f}%")
        ax.text(0.02, 0.4, stats_text, transform=ax.transAxes, fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
        
        plt.savefig(os.path.join(OUTPUT_DIR, "rolling_backtest_performance.png"))
        print(f"\n✅ Rolling backtest complete. Chart saved to '{OUTPUT_DIR}'.")