import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import warnings
import random
import itertools
import os

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import gymnasium as gym
from torch.distributions import Categorical, Normal
import yfinance as yf
import pandas_ta as ta

warnings.filterwarnings("ignore")

# --- VERSION CUE ---
SCRIPT_VERSION = "Portfolio Backtester v3.1 (Rolling Supervised ATR)"

def set_seeds(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# --- DATA, FEATURE, AND AGENT CODE ---
def fetch_and_prepare_data(tickers=["QQQ"], period="729d", interval="1h"):
    print(f"--- Fetching and Preparing Multi-Scale Data for {', '.join(tickers)} ---")
    vix_df = yf.Ticker("^VIX").history(period=period, interval="1d", auto_adjust=False)
    if not vix_df.empty:
        vix_df.index = vix_df.index.tz_convert('UTC')
        vix_df.rename(columns={'Close': 'vix'}, inplace=True)
    all_asset_data = {}
    for ticker in tickers:
        base_df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
        if base_df.empty: continue
        if vix_df.empty: base_df['vix'] = 20
        else:
            base_df.index = base_df.index.tz_convert('UTC')
            base_df = pd.merge_asof(left=base_df.sort_index(), right=vix_df[['vix']].sort_index(), left_index=True, right_index=True, direction='backward')
        agg_logic = {'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum','vix':'last'}
        df_1d = base_df.resample('D').agg(agg_logic).dropna(); df_1w = base_df.resample('W-MON').agg(agg_logic).dropna()
        dataframes = {'1h': base_df, '1d': df_1d, '1w': df_1w}; processed_dfs = {}
        for timeframe, df in dataframes.items():
            df_processed = df.copy(); df_processed['returns'] = (df_processed['Close']-df_processed['Open'])/df_processed['Open']
            df_processed.ta.rsi(length=14, append=True); df_processed.ta.atr(length=14, append=True); df_processed.ta.macd(append=True); df_processed.ta.adx(append=True)
            df_processed.rename(columns={"RSI_14":"rsi", "ATRr_14":"atr"}, inplace=True); df_processed['rsi'] = df_processed['rsi'] / 100.0; df_processed.dropna(inplace=True)
            required_cols = ['Open','High','Low','Close','Volume','returns','rsi','atr','vix', 'MACD_12_26_9','MACDh_12_26_9','MACDs_12_26_9','ADX_14']
            processed_dfs[timeframe] = df_processed[required_cols]
        all_asset_data[ticker] = processed_dfs
    return all_asset_data

class BayesianAnomalyDetector:
    def __init__(self, distribution_type='t'): self.dist_type = distribution_type; self.alpha_0, self.beta_0, self.mu_0, self.nu_0 = 1.0, 1.0, 0.0, 1.0; self.reset_posterior()
    def reset_posterior(self): self.alpha_n, self.beta_n, self.mu_n, self.nu_n = self.alpha_0, self.beta_0, self.mu_0, self.nu_0
    def fit(self, data):
        if self.dist_type == 't': self._fit_t(data)
    def compute_surprise(self, x):
        from scipy.stats import t
        df,loc,scale=2*self.alpha_n,self.mu_n,np.sqrt(self.beta_n/(self.alpha_n*self.nu_n));
        if scale <= 0: return 0.5
        return 1.0 - (2 * min(t.cdf(x,df=df,loc=loc,scale=scale), t.sf(x,df=df,loc=loc,scale=scale)))
    def get_distribution_params(self): return self.mu_n, np.sqrt(self.beta_n/(self.alpha_n*self.nu_n))
    def _fit_t(self, data):
        n=len(data);
        if n==0: return
        mean_data,sum_sq_diff=data.mean(),((data-data.mean())**2).sum()
        self.alpha_n=self.alpha_0+n/2; self.beta_n=self.beta_0+0.5*sum_sq_diff+(n*self.nu_0)/(self.nu_0+n)*0.5*(mean_data-self.mu_0)**2
        self.mu_n=(self.nu_0*self.mu_0+n*mean_data)/(self.nu_0+n); self.nu_n=self.nu_0+n

class RollingNormalizer:
    def __init__(self, window_size=252): self.window_size, self.data, self.min, self.max = window_size, deque(maxlen=window_size), None, None
    def update(self, value): self.data.append(value); self.min, self.max = min(self.data), max(self.data)
    def normalize(self, value): return 0.5 if self.max is None or self.min is None or self.max == self.min else (value - self.min) / (self.max - self.min)

def calculate_walk_forward_features(all_asset_data):
    all_features_dict = {}
    aligned_dfs_dict = {}
    for ticker, dataframes in all_asset_data.items():
        timeframes=['1h','1d','1w']; indicators=['returns','Volume','rsi','atr','vix','MACD_12_26_9','MACDh_12_26_9','MACDs_12_26_9','ADX_14']
        window_sizes={'1h':882,'1d':126,'1w':26}; detectors,normalizers,data_windows={},{},{}
        for tf,ind in itertools.product(timeframes,indicators):
            key=f'{tf}_{ind}';win_size=window_sizes[tf]
            detectors[key]=BayesianAnomalyDetector(distribution_type='t'); normalizers[f'{key}_raw']=RollingNormalizer(win_size)
            normalizers[f'{key}_mean']=RollingNormalizer(win_size); normalizers[f'{key}_scale']=RollingNormalizer(win_size); data_windows[key]=list(dataframes[tf][ind].iloc[:win_size])
        for key,window in data_windows.items():
            detectors[key].fit(pd.Series(window)); [normalizers[f'{key}_raw'].update(val) for val in window]
        asset_features=[]; start_ts=dataframes['1w'].index[window_sizes['1w']]; start_index=dataframes['1h'].index.searchsorted(start_ts)
        for i in tqdm(range(start_index,len(dataframes['1h'])), desc=f"  Calculating {ticker} Features"):
            step_features=[]; current_ts=dataframes['1h'].index[i]
            for tf in timeframes:
                tf_loc=dataframes[tf].index.searchsorted(current_ts,side='right')-1; current_observation=dataframes[tf].iloc[tf_loc]
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
        all_features_dict[ticker] = np.array(asset_features,dtype=np.float32)
        aligned_dfs_dict[ticker] = dataframes['1h'].iloc[start_index:start_index+len(asset_features)]
    return all_features_dict, aligned_dfs_dict

class TradingEnvironment(gym.Env):
    def __init__(self, df, surprise_scores, initial_balance=100000, trade_fee=0.001, reward_scheme='sortino'):
        super().__init__(); self.df = df.iloc[-len(surprise_scores):].reset_index(drop=True); self.features = surprise_scores
        self.initial_balance = initial_balance; self.trade_fee = trade_fee; self.reward_scheme = reward_scheme
        self.action_space = gym.spaces.Tuple((gym.spaces.Discrete(3), gym.spaces.Box(low=np.array([1.0, 0.0, 0.0]), high=np.array([3.0, 0.1, 0.2]), dtype=np.float32)))
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.features.shape[1],), dtype=np.float32)
        self.portfolio_history = deque(maxlen=22)
    def reset(self, seed=None):
        super().reset(seed=seed); self._current_step = 0
        self.balance, self.shares_held, self.position, self.entry_price, self.stop_loss_price = self.initial_balance, 0, 0, 0, 0
        self.portfolio_value = self.balance; self.returns_history = []; self.portfolio_history.clear(); self.portfolio_history.append(self.initial_balance)
        self.done = False
        return self.features[0], {'portfolio_value': self.portfolio_value}
    def step(self, action):
        discrete_action, continuous_params = action; stop_loss_atr_multiplier, trend_reward_bonus, trade_penalty = continuous_params
        current_price, current_atr = self.df.iloc[self._current_step]['Close'], self.df.iloc[self._current_step]['atr']
        old_portfolio_value = self.portfolio_value; trade_executed = False
        if self.position == 1 and current_price < self.stop_loss_price: discrete_action = 2
        elif self.position == -1 and current_price > self.stop_loss_price: discrete_action = 1
        if discrete_action == 1:
            if self.position == -1: self.balance=self.portfolio_value*(1-self.trade_fee); self.position,self.shares_held,self.entry_price=0,0,0; trade_executed=True
            elif self.position == 0: self.position,self.entry_price=1,current_price; self.stop_loss_price=current_price-(current_atr*stop_loss_atr_multiplier); self.shares_held=self.balance/current_price*(1-self.trade_fee); trade_executed=True
        elif discrete_action == 2:
            if self.position == 1: self.balance=self.shares_held*current_price*(1-self.trade_fee); self.position,self.shares_held,self.entry_price=0,0,0; trade_executed=True
            elif self.position == 0: self.position,self.entry_price=-1,current_price; self.stop_loss_price=current_price+(current_atr*stop_loss_atr_multiplier); self.shares_held=self.balance/current_price*(1-self.trade_fee); trade_executed=True
        self.portfolio_value = self.shares_held*current_price if self.position==1 else (self.balance+(self.entry_price-current_price)*self.shares_held if self.position==-1 else self.balance)
        self.returns_history.append((self.portfolio_value/old_portfolio_value)-1 if old_portfolio_value!=0 else 0)
        self.portfolio_history.append(self.portfolio_value)
        reward = 0
        if len(self.returns_history) > 21:
            relevant_returns = np.array(self.returns_history[-21:]); mean_return = np.mean(relevant_returns)
            if self.reward_scheme == 'calmar':
                annualized_return = mean_return * (252 * 7); portfolio_values = np.array(self.portfolio_history); running_max = np.maximum.accumulate(portfolio_values)
                drawdowns = (running_max - portfolio_values) / running_max; max_drawdown = np.max(drawdowns)
                if max_drawdown > 0: reward = np.tanh(annualized_return / max_drawdown)
                elif mean_return > 0: reward = 1.0
            else:
                std_dev_returns = np.std(relevant_returns)
                if std_dev_returns != 0: reward = np.tanh(mean_return / std_dev_returns * np.sqrt(252*7))
                elif mean_return > 0: reward = 1.0
        if trade_executed: reward -= trade_penalty
        if (self.position==1 and current_price>self.df.iloc[self._current_step]['Open']) or (self.position==-1 and current_price<self.df.iloc[self._current_step]['Open']): reward += trend_reward_bonus
        self._current_step+=1
        if self._current_step >= len(self.features): self.done = True
        return (self.features[self._current_step] if not self.done else self.features[-1]), reward, self.done, False, {'portfolio_value': self.portfolio_value}

class RolloutBuffer:
    def __init__(self): self.actions,self.states,self.logprobs,self.rewards,self.is_terminals,self.values=[],[],[],[],[],[]
    def clear(self): del self.actions[:],self.states[:],self.logprobs[:],self.rewards[:],self.is_terminals[:],self.values[:]

class ActorCritic(nn.Module):
    def __init__(self, n_observations, n_discrete_actions, n_continuous_actions):
        super(ActorCritic, self).__init__(); self.shared_body=nn.Sequential(nn.Linear(n_observations,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU())
        self.policy_head, self.param_head, self.value_head = nn.Linear(256, n_discrete_actions), nn.Linear(256, n_continuous_actions), nn.Linear(256, 1)
        self.log_std_head=nn.Parameter(torch.zeros(n_continuous_actions))
    def forward(self, x):
        x=self.shared_body(x); return torch.softmax(self.policy_head(x),dim=-1), torch.tanh(self.param_head(x)), torch.exp(self.log_std_head), self.value_head(x)

class PPOAgent:
    def __init__(self, env, lr=3e-4, entropy_coef=0.01):
        self.env=env; self.obs_dim, self.disc_action_dim, self.cont_action_dim = env.observation_space.shape[0], env.action_space[0].n, env.action_space[1].shape[0]
        self.cont_action_low, self.cont_action_high = torch.tensor(env.action_space[1].low), torch.tensor(env.action_space[1].high)
        self.lr, self.gamma, self.gae_lambda = lr, 0.99, 0.95; self.clip_epsilon, self.entropy_coef, self.value_loss_coef = 0.2, entropy_coef, 0.5
        self.epochs, self.mini_batch_size = 10, 64; self.policy = ActorCritic(self.obs_dim, self.disc_action_dim, self.cont_action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(),lr=self.lr); self.buffer = RolloutBuffer()
    def _scale_continuous_action(self,tanh_action): return self.cont_action_low+(tanh_action+1.0)*0.5*(self.cont_action_high-self.cont_action_low)
    def select_action(self,state):
        with torch.no_grad():
            state_tensor=torch.FloatTensor(state).unsqueeze(0); action_probs,means,stds,value=self.policy(state_tensor)
            cat_dist, norm_dist = Categorical(action_probs), Normal(means,stds)
            discrete_action, continuous_actions_tanh = cat_dist.sample(), norm_dist.sample()
            total_log_prob = cat_dist.log_prob(discrete_action) + norm_dist.log_prob(continuous_actions_tanh).sum(dim=-1)
            action=(discrete_action.item(), self._scale_continuous_action(continuous_actions_tanh).squeeze(0).numpy())
        return state,action,total_log_prob.item(),value.item()
    def learn(self):
        rewards, is_terminals, values = torch.tensor(self.buffer.rewards,dtype=torch.float32), torch.tensor(self.buffer.is_terminals,dtype=torch.float32), torch.tensor(self.buffer.values,dtype=torch.float32)
        advantages = torch.zeros_like(rewards); last_advantage=0
        for t in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - is_terminals[t] if t==len(rewards)-1 else 1.0 - is_terminals[t+1]; next_value = 0 if t==len(rewards)-1 else values[t+1]
            delta = rewards[t] + self.gamma*next_value*next_non_terminal - values[t]
            advantages[t]=last_advantage=delta+self.gamma*self.gae_lambda*next_non_terminal*last_advantage
        returns = advantages + values; advantages = (advantages - advantages.mean())/(advantages.std()+1e-8)
        old_states = torch.FloatTensor(np.array(self.buffer.states)); old_discrete_actions = torch.tensor([a[0] for a in self.buffer.actions],dtype=torch.int64); old_tanh_actions = torch.tensor(np.array([a[1] for a in self.buffer.actions]),dtype=torch.float32); old_logprobs = torch.tensor(self.buffer.logprobs,dtype=torch.float32)
        for _ in range(self.epochs):
            for index in torch.randperm(len(old_states)).split(self.mini_batch_size):
                action_probs,means,stds,state_values=self.policy(old_states[index]); cat_dist, norm_dist = Categorical(action_probs), Normal(means,stds)
                logprobs = cat_dist.log_prob(old_discrete_actions[index])+norm_dist.log_prob(old_tanh_actions[index]).sum(dim=-1)
                dist_entropy = cat_dist.entropy()+norm_dist.entropy().sum(dim=-1); ratios = torch.exp(logprobs-old_logprobs[index])
                surr1, surr2 = ratios*advantages[index], torch.clamp(ratios,1-self.clip_epsilon,1+self.clip_epsilon)*advantages[index]
                policy_loss, value_loss, entropy_loss = -torch.min(surr1,surr2).mean(), nn.MSELoss()(state_values.squeeze(),returns[index]), -dist_entropy.mean()
                loss = policy_loss+self.value_loss_coef*value_loss+self.entropy_coef*entropy_loss
                self.optimizer.zero_grad();loss.backward();self.optimizer.step()
        self.buffer.clear()

def train_agent_on_window(train_df, train_features, total_timesteps, params, pretrained_policy=None, seed=42, reward_scheme='sortino'):
    env = TradingEnvironment(df=train_df, surprise_scores=train_features, reward_scheme=reward_scheme)
    agent = PPOAgent(env, lr=params.get('lr', 3e-4), entropy_coef=params.get('entropy_coef', 0.01))
    if pretrained_policy:
        agent.policy.load_state_dict(pretrained_policy)
    state, _ = env.reset(seed=seed)
    for timestep in range(total_timesteps):
        old_state, action, log_prob, value = agent.select_action(state)
        next_state, reward, done, _, _ = env.step(action)
        agent.buffer.states.append(old_state);agent.buffer.actions.append(action);agent.buffer.logprobs.append(log_prob)
        agent.buffer.rewards.append(reward);agent.buffer.is_terminals.append(done);agent.buffer.values.append(value)
        state=next_state
        if done: agent.learn(); state, _ = env.reset(seed=seed + timestep + 1)
    return agent.policy.state_dict()

def test_agent_on_window(policy_state_dict, test_df, test_features, initial_balance, seed=42):
    test_env = TradingEnvironment(df=test_df, surprise_scores=test_features, initial_balance=initial_balance)
    obs_dim, disc_action_dim, cont_action_dim = test_env.observation_space.shape[0], test_env.action_space[0].n, test_env.action_space[1].shape[0]
    policy_net = ActorCritic(obs_dim, disc_action_dim, cont_action_dim); policy_net.load_state_dict(policy_state_dict); policy_net.eval()
    state, _ = test_env.reset(seed=seed); log = [{'Timestamp': test_df.index[0] - pd.Timedelta(hours=1), 'PortfolioValue': initial_balance}]; done=False; cs=0
    while not done:
        st = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            ap,mn,_,_ = policy_net(st); da=torch.argmax(ap).item()
            low,high=torch.tensor(test_env.action_space[1].low),torch.tensor(test_env.action_space[1].high)
            cas=low+(mn+1.0)*0.5*(high-low)
            act=(da, cas.squeeze(0).numpy())
        state, _, done, _, info = test_env.step(act)
        log.append({'Timestamp': test_df.index[cs], 'PortfolioValue': info['portfolio_value']}); cs += 1
    return pd.DataFrame(log)

def generate_atr_regime_labels_quantile(df, bull_quantile=0.4, bear_quantile=0.8):
    bull_threshold = df['atr'].quantile(bull_quantile)
    bear_threshold = df['atr'].quantile(bear_quantile)
    labels = pd.Series('SIDEWAYS', index=df.index, dtype=str)
    labels[df['atr'] < bull_threshold] = 'BULL'
    labels[df['atr'] > bear_threshold] = 'BEAR'
    return labels

def run_portfolio_walk_forward(tickers, features_dict, dfs_dict, total_training_steps, params_dict, initial_portfolio_balance=100000, save_plot_filename=None, seed=42):
    train_ws, test_ws = 252*7, 21*7
    
    num_assets = len(tickers); initial_bot_balance = initial_portfolio_balance / num_assets
    current_balances = {ticker: initial_bot_balance for ticker in tickers}
    portfolio_policies = {ticker: {'BULL': None, 'BEAR': None, 'SIDEWAYS': None} for ticker in tickers}
    all_portfolio_results = []
    min_len = min(len(features) for features in features_dict.values())
    num_w = (min_len - train_ws) // test_ws
    
    curated_indices = [ (0*36)+(0*4)+1, (1*36)+(5*4)+1, (0*36)+(4*4)+1, (0*36)+(3*4)+1, (1*36)+(8*4)+1, (1*36)+(2*4)+1 ]

    print(f"\n--- Starting Multi-Asset Portfolio Backtest ({num_w} windows) for Seed {seed} ---")
    
    for i in range(num_w):
        print(f"\n--- Window {i+1}/{num_w} ---")
        window_results = []
        for ticker in tickers:
            print(f"  -- Processing Asset: {ticker} --")
            full_df, all_features = dfs_dict[ticker], features_dict[ticker]
            train_si, train_ei = i * test_ws, i * test_ws + train_ws
            
            train_df_window = full_df.iloc[train_si:train_ei]
            train_features_window = all_features[train_si:train_ei]
            X_window = train_features_window[:, curated_indices]
            y_window = generate_atr_regime_labels_quantile(train_df_window)
            
            if len(set(y_window)) < 2:
                mapped_label = 'SIDEWAYS'
            else:
                scaler = StandardScaler().fit(X_window)
                X_window_scaled = scaler.transform(X_window)
                model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=200).fit(X_window_scaled, y_window)
                window_predictions = model.predict(X_window_scaled)
                mapped_label = pd.Series(window_predictions).mode()[0] if len(window_predictions) > 0 else 'SIDEWAYS'
            
            rew_sch = 'sharpe' if mapped_label == 'BULL' else 'calmar'
            print(f"  Regime for {ticker}: '{mapped_label}'. Using '{rew_sch}' reward.")
            
            active_specialist_policy = portfolio_policies[ticker][mapped_label]
            current_params = params_dict[ticker].get(mapped_label, {})
            
            trained_policy = train_agent_on_window(train_df_window, train_features_window, total_training_steps, current_params, active_specialist_policy, seed + i, rew_sch)
            portfolio_policies[ticker][mapped_label] = trained_policy
            
            test_si, test_ei = train_ei, train_ei + test_ws
            if test_ei > len(all_features): continue
            test_df, test_feat = full_df.iloc[test_si:test_ei], all_features[test_si:test_ei]
            
            bot_results_df = test_agent_on_window(trained_policy, test_df, test_feat, current_balances[ticker], seed + i)
            if not bot_results_df.empty:
                current_balances[ticker] = bot_results_df['PortfolioValue'].iloc[-1]
                bot_window_df = bot_results_df.set_index('Timestamp')[['PortfolioValue']].rename(columns={'PortfolioValue': ticker})
                window_results.append(bot_window_df)
        
        if window_results:
            combined_window_df = pd.concat(window_results, axis=1).ffill().bfill()
            all_portfolio_results.append(combined_window_df)
            
    if not all_portfolio_results: return pd.DataFrame()
    
    final_results_df = pd.concat(all_portfolio_results)
    final_results_df['TotalPortfolioValue'] = final_results_df.sum(axis=1)
    
    if save_plot_filename:
        start_row = pd.DataFrame({'TotalPortfolioValue': initial_portfolio_balance}, index=[final_results_df.index[0] - pd.Timedelta(hours=1)])
        plot_df = pd.concat([start_row, final_results_df])
        benchmark_df = pd.DataFrame(index=plot_df.index)
        for ticker in tickers:
            asset_prices = dfs_dict[ticker]['Close'].reindex(plot_df.index, method='ffill').bfill()
            initial_shares = initial_bot_balance / asset_prices.iloc[0]
            benchmark_df[ticker] = asset_prices * initial_shares
        benchmark_df['TotalPortfolioValue'] = benchmark_df.sum(axis=1)

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(plot_df.index, plot_df['TotalPortfolioValue'], label=f'Agent Portfolio (Seed {seed})', color='deepskyblue', linewidth=2)
        ax.plot(benchmark_df.index, benchmark_df['TotalPortfolioValue'], label='Equal-Weight Buy & Hold', color='gray', linestyle='--', linewidth=1.5)
        ax.set_title(f"Baseline Portfolio Performance (Seed {seed})")
        ax.set_xlabel("Date"); ax.set_ylabel("Portfolio Value ($)")
        ax.legend(); fig.tight_layout()
        plt.savefig(save_plot_filename)
        plt.close()

    return final_results_df

if __name__ == '__main__':
    print(f"--- {SCRIPT_VERSION} ---")
    
    BASELINE_PORTFOLIO = ["QQQ", "GLD", "TLT"]
    INITIAL_CAPITAL = 100000
    TOTAL_TRAINING_STEPS_PER_WINDOW = 50000
    VERIFICATION_SEEDS = range(101, 111)
    
    # --- DEFINITIVE ROBUST CHAMPION HYPERPARAMETERS ---
    CHAMPION_PARAMS = {
        "QQQ": {
            'BULL':     {'lr': 0.000070, 'entropy_coef': 0.0484},
            'BEAR':     {'lr': 0.000019, 'entropy_coef': 0.0007},
            'SIDEWAYS': {'lr': 0.000012, 'entropy_coef': 0.0562}
        },
        "GLD": {
            'BULL':     {'lr': 0.001000, 'entropy_coef': 0.0000},
            'BEAR':     {'lr': 0.000010, 'entropy_coef': 0.0000},
            'SIDEWAYS': {'lr': 0.000100, 'entropy_coef': 0.0000}
        },
        "TLT": {
            'BULL':     {'lr': 0.000261, 'entropy_coef': 0.0367},
            'BEAR':     {'lr': 0.000440, 'entropy_coef': 0.0817},
            'SIDEWAYS': {'lr': 0.000010, 'entropy_coef': 0.1000}
        }
    }
    
    print("\n--- Starting 10-Seed Verification for Baseline Portfolio ---")
    os.makedirs('baseline_runs', exist_ok=True)
    
    all_data = fetch_and_prepare_data(tickers=BASELINE_PORTFOLIO, period="729d", interval="1h")
    if not all_data or len(all_data) != len(BASELINE_PORTFOLIO):
        print("Could not fetch all necessary data. Exiting."); exit()
    features_dict, dfs_dict = calculate_walk_forward_features(all_data)
    
    final_values = []
    for seed in VERIFICATION_SEEDS:
        set_seeds(seed)
        print(f"\n===== RUNNING VERIFICATION FOR SEED: {seed} =====")
        results_df = run_portfolio_walk_forward(
            tickers=BASELINE_PORTFOLIO, features_dict=features_dict, dfs_dict=dfs_dict,
            total_training_steps=TOTAL_TRAINING_STEPS_PER_WINDOW, params_dict=CHAMPION_PARAMS,
            initial_portfolio_balance=INITIAL_CAPITAL,
            save_plot_filename=f'baseline_runs/equity_curve_seed_{seed}.png', seed=seed
        )
        if results_df is not None and not results_df.empty:
            final_value = results_df['TotalPortfolioValue'].iloc[-1]
            final_values.append(final_value)
            print(f"===== SEED {seed} COMPLETE: FINAL VALUE = ${final_value:,.2f} =====")

    if final_values:
        mean_perf = np.mean(final_values)
        std_perf = np.std(final_values)
        min_perf = np.min(final_values)
        max_perf = np.max(final_values)
        print("\n\n--- BASELINE PORTFOLIO 10-SEED VERIFICATION REPORT ---")
        print(f"Portfolio: {', '.join(BASELINE_PORTFOLIO)}")
        print("-" * 50)
        print(f"Mean Final Value:      ${mean_perf:,.2f}")
        print(f"Standard Deviation:    ${std_perf:,.2f}")
        print(f"Best Performance (Max):${max_perf:,.2f}")
        print(f"Worst Performance (Min):${min_perf:,.2f}")
        print("-" * 50)