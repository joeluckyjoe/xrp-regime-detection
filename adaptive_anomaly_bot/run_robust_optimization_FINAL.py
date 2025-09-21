import pandas as pd
import numpy as np
import warnings
import random
import torch
from tqdm import tqdm
import os

from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

import gymnasium as gym
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
import yfinance as yf
import pandas_ta as ta
from collections import deque
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# --- VERSION CUE ---
SCRIPT_VERSION = "Optimizer v3.0 (Rolling Supervised ATR)"

def set_seeds(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# --- DATA, FEATURE, AND AGENT CODE (Full, Corrected Versions) ---
def fetch_and_prepare_data(ticker="QQQ", period="729d", interval="1h"):
    base_df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
    if base_df.empty: return None
    vix_df = yf.Ticker("^VIX").history(period=period, interval="1d", auto_adjust=False)
    if not vix_df.empty:
        vix_df.index = vix_df.index.tz_convert('UTC')
        vix_df.rename(columns={'Close': 'vix'}, inplace=True)
        base_df.index = base_df.index.tz_convert('UTC')
        base_df = pd.merge_asof(left=base_df.sort_index(), right=vix_df[['vix']].sort_index(), left_index=True, right_index=True, direction='backward')
    else: base_df['vix'] = 20
    agg_logic = {'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum','vix':'last'}
    df_1d = base_df.resample('D').agg(agg_logic).dropna(); df_1w = base_df.resample('W-MON').agg(agg_logic).dropna()
    dataframes = {'1h': base_df, '1d': df_1d, '1w': df_1w}; processed_dfs = {}
    for timeframe, df in dataframes.items():
        df_processed = df.copy(); df_processed['returns'] = (df_processed['Close']-df_processed['Open'])/df_processed['Open']
        df_processed.ta.rsi(length=14, append=True); df_processed.ta.atr(length=14, append=True); df_processed.ta.macd(append=True); df_processed.ta.adx(append=True)
        df_processed.rename(columns={"RSI_14":"rsi", "ATRr_14":"atr"}, inplace=True); df_processed['rsi'] = df_processed['rsi'] / 100.0; df_processed.dropna(inplace=True)
        required_cols = ['Open','High','Low','Close','Volume','returns','rsi','atr','vix', 'MACD_12_26_9','MACDh_12_26_9','MACDs_12_26_9','ADX_14']
        processed_dfs[timeframe] = df_processed[required_cols]
    return processed_dfs

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

def calculate_walk_forward_features(dataframes):
    timeframes=['1h','1d','1w']; indicators=['returns','Volume','rsi','atr','vix','MACD_12_26_9','MACDh_12_26_9','MACDs_12_26_9','ADX_14']
    window_sizes={'1h':882,'1d':126,'1w':26}; detectors,normalizers,data_windows={},{},{}
    for tf,ind in itertools.product(timeframes,indicators):
        key=f'{tf}_{ind}';win_size=window_sizes[tf]
        detectors[key]=BayesianAnomalyDetector(distribution_type='t'); normalizers[f'{key}_raw']=RollingNormalizer(win_size)
        normalizers[f'{key}_mean']=RollingNormalizer(win_size); normalizers[f'{key}_scale']=RollingNormalizer(win_size); data_windows[key]=list(dataframes[tf][ind].iloc[:win_size])
    for key,window in data_windows.items():
        detectors[key].fit(pd.Series(window)); [normalizers[f'{key}_raw'].update(val) for val in window]
    all_features=[]; start_ts=dataframes['1w'].index[window_sizes['1w']]; start_index=dataframes['1h'].index.searchsorted(start_ts)
    for i in tqdm(range(start_index,len(dataframes['1h'])), desc="  Calculating Features", leave=False, ncols=80):
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
        if len(step_features)==108: all_features.append(step_features)
    return np.array(all_features,dtype=np.float32), dataframes['1h'].iloc[start_index:start_index+len(all_features)]

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
    if pretrained_policy: agent.policy.load_state_dict(pretrained_policy)
    state, _ = env.reset(seed=seed)
    for timestep in range(total_timesteps):
        old_state, action, log_prob, value = agent.select_action(state); next_state, reward, done, _, _ = env.step(action)
        agent.buffer.states.append(old_state);agent.buffer.actions.append(action);agent.buffer.logprobs.append(log_prob);agent.buffer.rewards.append(reward);agent.buffer.is_terminals.append(done);agent.buffer.values.append(value)
        state=next_state
        if done: agent.learn(); state, _ = env.reset(seed=seed + timestep + 1)
    return agent.policy.state_dict()

def test_agent_on_window(policy_state_dict, test_df, test_features, initial_balance, seed=42):
    test_env = TradingEnvironment(df=test_df, surprise_scores=test_features, initial_balance=initial_balance)
    obs_dim, disc_action_dim, cont_action_dim = test_env.observation_space.shape[0], test_env.action_space[0].n, test_env.action_space[1].shape[0]
    policy_net = ActorCritic(obs_dim, disc_action_dim, cont_action_dim); policy_net.load_state_dict(policy_state_dict); policy_net.eval()
    state, _ = test_env.reset(seed=seed); log = [{'PortfolioValue': initial_balance}]; done=False
    while not done:
        st = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            ap,mn,_,_ = policy_net(st); da=torch.argmax(ap).item()
            low,high=torch.tensor(test_env.action_space[1].low),torch.tensor(test_env.action_space[1].high)
            cas=low+(mn+1.0)*0.5*(high-low)
            act=(da, cas.squeeze(0).numpy())
        state, _, done, _, info = test_env.step(act); log.append({'PortfolioValue': info['portfolio_value']})
    return pd.DataFrame(log)

# --- DEFINITIVE BACKTESTING FUNCTION WITH ROLLING SUPERVISED (ATR) DETECTOR ---
def generate_atr_regime_labels_quantile(df, bull_quantile=0.4, bear_quantile=0.8):
    bull_threshold = df['atr'].quantile(bull_quantile)
    bear_threshold = df['atr'].quantile(bear_quantile)
    labels = pd.Series('SIDEWAYS', index=df.index, dtype=str)
    labels[df['atr'] < bull_threshold] = 'BULL'
    labels[df['atr'] > bear_threshold] = 'BEAR'
    return labels

def run_walk_forward_analysis(full_df, all_features, total_training_steps, params, seed=42):
    train_ws, test_ws = 252*7, 21*7
    init_b = 100000; curr_b = init_b; all_res = []
    num_w = (len(all_features) - train_ws) // test_ws
    spec_pols = {'BULL': None, 'BEAR': None, 'SIDEWAYS': None}
    curated_indices = [ (0*36)+(0*4)+1, (1*36)+(5*4)+1, (0*36)+(4*4)+1, (0*36)+(3*4)+1, (1*36)+(8*4)+1, (1*36)+(2*4)+1 ]
    
    for i in tqdm(range(num_w), desc=f"  Backtest Seed {seed}", leave=False, ncols=100):
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
        current_params = params.get(mapped_label, {})
        
        trained_pol_dict = train_agent_on_window(train_df_window, train_features_window, total_training_steps, current_params, spec_pols[mapped_label], seed + i, rew_sch)
        spec_pols[mapped_label] = trained_pol_dict
        
        test_si, test_ei = train_ei, train_ei + test_ws
        if test_ei > len(all_features): break
        test_df, test_feat = full_df.iloc[test_si:test_ei], all_features[test_si:test_ei]
        month_res = test_agent_on_window(trained_pol_dict, test_df, test_feat, curr_b, seed + i)
        if not month_res.empty:
            all_res.append(month_res.iloc[1:])
            curr_b = month_res['PortfolioValue'].iloc[-1]
            
    if not all_res: return pd.DataFrame([{'PortfolioValue': init_b}])
    final_df = pd.concat(all_res, ignore_index=True)
    return pd.concat([pd.DataFrame([{'PortfolioValue': init_b}]), final_df], ignore_index=True)

# --- ROBUST OPTIMIZATION SCRIPT ---
SEARCH_SPACE = [
    Real(low=1e-5, high=1e-3, prior='log-uniform', name='lr_bull'),
    Real(low=0.0, high=0.1, name='entropy_coef_bull'),
    Real(low=1e-5, high=1e-3, prior='log-uniform', name='lr_bear'),
    Real(low=0.0, high=0.1, name='entropy_coef_bear'),
    Real(low=1e-6, high=1e-4, prior='log-uniform', name='lr_sideways'),
    Real(low=0.0, high=0.1, name='entropy_coef_sideways')
]
OPTIMIZATION_SEEDS = [42, 43, 44]
ASSET_TICKER = "TLT" # <--- CHANGE THIS FOR EACH ASSET
LOG_FILE_BASE = 'robust_optimization_log'
N_CALLS = 50

LOG_FILE = f'{LOG_FILE_BASE}_{ASSET_TICKER}_FINAL.csv'
DETAILED_LOG_FILE = f'{LOG_FILE_BASE}_DETAILED_{ASSET_TICKER}_FINAL.csv'

print(f"--- {SCRIPT_VERSION} ---")
print(f"\n--- Pre-loading and preparing data for {ASSET_TICKER} ---")
all_data_dfs = fetch_and_prepare_data(ticker=ASSET_TICKER)
if all_data_dfs:
    all_features, aligned_df = calculate_walk_forward_features(all_data_dfs)
    print("--- Data preparation complete. Ready for robust optimization. ---\n")
else:
    print(f"CRITICAL: Could not fetch data for {ASSET_TICKER}. Exiting."); exit()

RUN_COUNTER = 0
BEST_LCB_SCORE = -float('inf')

@use_named_args(SEARCH_SPACE)
def objective_function(lr_bull, entropy_coef_bull, lr_bear, entropy_coef_bear, lr_sideways, entropy_coef_sideways):
    global RUN_COUNTER, BEST_LCB_SCORE
    RUN_COUNTER += 1
    print(f"\n--- ROBUST OPTIMIZATION RUN {RUN_COUNTER}/{N_CALLS} ---")
    params = {'BULL': {'lr': lr_bull, 'entropy_coef': entropy_coef_bull},'BEAR': {'lr': lr_bear, 'entropy_coef': entropy_coef_bear},'SIDEWAYS': {'lr': lr_sideways, 'entropy_coef': entropy_coef_sideways}}
    print("Testing Parameters:"); [print(f"  - {r}: lr={p['lr']:.6f}, entropy_coef={p['entropy_coef']:.4f}") for r,p in params.items()]
    
    run_scores = []
    for seed in OPTIMIZATION_SEEDS:
        results_df = run_walk_forward_analysis(aligned_df.copy(), all_features.copy(), 50000, params, seed)
        final_value = results_df['PortfolioValue'].iloc[-1] if not results_df.empty else 0
        run_scores.append(final_value)
        print(f"    Seed {seed}: Final Value = ${final_value:,.2f}")
    
    mean_score, std_dev_score = np.mean(run_scores), np.std(run_scores)
    lcb_score = mean_score - std_dev_score
    
    print(f"\n===> Mean: ${mean_score:,.2f}, Std Dev: ${std_dev_score:,.2f}")
    print(f"===> LCB Score for Run {RUN_COUNTER}: ${lcb_score:,.2f}")

    if lcb_score > BEST_LCB_SCORE: BEST_LCB_SCORE = lcb_score; print(f"*** New Best LCB Score! ***")
    
    with open(LOG_FILE, 'a') as f:
        f.write(f"{lr_bull},{entropy_coef_bull},{lr_bear},{entropy_coef_bear},{lr_sideways},{entropy_coef_sideways},{lcb_score},{mean_score},{std_dev_score}\n")
    
    with open(DETAILED_LOG_FILE, 'a') as f:
        for i, score in enumerate(run_scores):
            f.write(f"{lr_bull},{entropy_coef_bull},{lr_bear},{entropy_coef_bear},{lr_sideways},{entropy_coef_sideways},{OPTIMIZATION_SEEDS[i]},{score}\n")

    return -lcb_score

if __name__ == '__main__':
    print(f"\n--- Bayesian Robust Optimization for {ASSET_TICKER} ---")
    
    x0_main, y0_main = None, None
    main_log_header = "lr_bull,entropy_coef_bull,lr_bear,entropy_coef_bear,lr_sideways,entropy_coef_sideways,lcb_score,mean_score,std_dev_score"
    if os.path.exists(LOG_FILE):
        print(f"Main log file '{LOG_FILE}' found. Attempting to resume.")
        try:
            log_df = pd.read_csv(LOG_FILE)
            if not log_df.empty:
                param_cols = ['lr_bull','entropy_coef_bull','lr_bear','entropy_coef_bear','lr_sideways','entropy_coef_sideways']
                if list(log_df.columns) != main_log_header.split(','): raise ValueError("Main log file header mismatch.")
                x0_main = log_df[param_cols].values.tolist(); y0_main = (-log_df['lcb_score']).values.tolist()
                RUN_COUNTER = len(x0_main)
                if 'lcb_score' in log_df.columns and not log_df['lcb_score'].empty: BEST_LCB_SCORE = log_df['lcb_score'].max()
                print(f"Loaded {len(x0_main)} previous trials. Current best LCB score: ${BEST_LCB_SCORE:,.2f}")
        except Exception as e:
            print(f"Error reading main log file: {e}. Starting new optimization."); os.rename(LOG_FILE, LOG_FILE + ".bak")
            with open(LOG_FILE, 'w') as f: f.write(main_log_header + "\n")
    else:
        print("Main log file not found. Starting new optimization.")
        with open(LOG_FILE, 'w') as f: f.write(main_log_header + "\n")
        
    detailed_log_header = "lr_bull,entropy_coef_bull,lr_bear,entropy_coef_bear,lr_sideways,entropy_coef_sideways,seed,final_value"
    if not os.path.exists(DETAILED_LOG_FILE):
        with open(DETAILED_LOG_FILE, 'w') as f: f.write(detailed_log_header + "\n")

    optimizer_kwargs = {"func": objective_function, "dimensions": SEARCH_SPACE, "n_calls": N_CALLS, "random_state": 123}
    if x0_main and y0_main:
        optimizer_kwargs['x0'], optimizer_kwargs['y0'] = x0_main, y0_main
        
    result = gp_minimize(**optimizer_kwargs)
    
    print("\n--- Robust Optimization Complete ---")
    print(f"Best parameters found for {ASSET_TICKER}:")
    best_params = {p.name: val for p, val in zip(SEARCH_SPACE, result.x)}
    print(f"  - BULL:     lr={best_params['lr_bull']:.6f}, entropy_coef={best_params['entropy_coef_bull']:.4f}")
    print(f"  - BEAR:     lr={best_params['lr_bear']:.6f}, entropy_coef={best_params['entropy_coef_bear']:.4f}")
    print(f"  - SIDEWAYS: lr={best_params['lr_sideways']:.6f}, entropy_coef={best_params['entropy_coef_sideways']:.4f}")
    print(f"Best LCB Score (Mean - Std Dev): ${-result.fun:,.2f}")