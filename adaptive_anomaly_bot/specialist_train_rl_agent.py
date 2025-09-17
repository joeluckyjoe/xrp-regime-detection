import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
import yfinance as yf
import pandas_ta as ta
import warnings
from collections import deque

warnings.filterwarnings("ignore", category=RuntimeWarning, module='scipy.stats._continuous_distns')

def fetch_and_prepare_data(tickers=["QQQ"], period="729d", interval="1h"):
    print(f"--- Fetching and Preparing Multi-Scale Data for {', '.join(tickers)} (v21 Multi-Asset) ---")
    
    # Fetch VIX data once, as it's common for all assets
    vix_df = yf.Ticker("^VIX").history(period=period, interval="1d", auto_adjust=False)
    vix_df.index = vix_df.index.tz_convert('UTC')
    vix_df.rename(columns={'Close': 'vix'}, inplace=True)
    
    all_asset_data = {}

    for ticker in tickers:
        print(f"  Processing {ticker}...")
        base_df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
        if base_df.empty:
            print(f"  WARNING: Could not fetch data for {ticker}. Skipping.")
            continue

        # Merge VIX data
        if vix_df.empty:
            base_df['vix'] = 20  # Fallback value if VIX fetch fails
        else:
            base_df.index = base_df.index.tz_convert('UTC')
            base_df = pd.merge_asof(
                left=base_df.sort_index(), 
                right=vix_df[['vix']].sort_index(), 
                left_index=True, 
                right_index=True, 
                direction='backward'
            )

        # Multi-scale aggregation
        agg_logic = {'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum', 'vix':'last'}
        df_1d = base_df.resample('D').agg(agg_logic).dropna()
        df_1w = base_df.resample('W-MON').agg(agg_logic).dropna()

        dataframes = {'1h': base_df, '1d': df_1d, '1w': df_1w}
        processed_dfs = {}

        # Feature engineering for each timeframe
        for timeframe, df in dataframes.items():
            df_processed = df.copy()
            df_processed['returns'] = (df_processed['Close'] - df_processed['Open']) / df_processed['Open']
            df_processed.ta.rsi(length=14, append=True)
            df_processed.ta.atr(length=14, append=True)
            df_processed.ta.macd(append=True)
            df_processed.ta.adx(append=True)
            df_processed.rename(columns={"RSI_14": "rsi", "ATRr_14": "atr"}, inplace=True)
            df_processed['rsi'] = df_processed['rsi'] / 100.0
            df_processed.dropna(inplace=True)
            
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'returns', 'rsi', 'atr', 'vix', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 'ADX_14']
            processed_dfs[timeframe] = df_processed[required_cols]
        
        all_asset_data[ticker] = processed_dfs

    print("--- Multi-Asset Data Preparation Complete ---")
    return all_asset_data

class BayesianAnomalyDetector:
    def __init__(self, distribution_type='t'): self.dist_type = distribution_type; self.alpha_0, self.beta_0, self.mu_0, self.nu_0 = 1.0, 1.0, 0.0, 1.0; self.reset_posterior()
    def reset_posterior(self): self.alpha_n, self.beta_n, self.mu_n, self.nu_n = self.alpha_0, self.beta_0, self.mu_0, self.nu_0; self.mle_params = None
    def fit(self, data):
        if self.dist_type == 't': self._fit_t(data)
        elif self.dist_type == 'beta': self._fit_beta(data)
        elif self.dist_type == 'gamma': self._fit_gamma(data)
    def compute_surprise(self, x):
        from scipy.stats import t, beta, gamma
        if self.mle_params is None and self.dist_type in ['beta', 'gamma']: return 0.5
        if self.dist_type == 't':
            df,loc,scale=2*self.alpha_n,self.mu_n,np.sqrt(self.beta_n/(self.alpha_n*self.nu_n));
            if scale <= 0: return 0.5
            return 1.0 - (2 * min(t.cdf(x,df=df,loc=loc,scale=scale), t.sf(x,df=df,loc=loc,scale=scale)))
        elif self.dist_type == 'beta':
            x_norm = max(0.0001, min(x, 0.9999)); a,b,loc,scale = self.mle_params
            return 1.0 - (2 * min(beta.cdf(x_norm,a,b,loc,scale), beta.sf(x_norm,a,b,loc,scale)))
        elif self.dist_type == 'gamma':
            a,loc,scale = self.mle_params
            if scale <= 0: return 0.5
            return 1.0 - (2 * min(gamma.cdf(x,a,loc=loc,scale=scale), gamma.sf(x,a,loc=loc,scale=scale)))
    def get_distribution_params(self):
        from scipy.stats import beta, gamma
        if self.dist_type=='t': return self.mu_n, np.sqrt(self.beta_n/(self.alpha_n*self.nu_n))
        elif self.dist_type=='beta' and self.mle_params: a,b,loc,scale=self.mle_params; return beta.mean(a,b,loc,scale), beta.std(a,b,loc,scale)
        elif self.dist_type=='gamma' and self.mle_params: a,loc,scale=self.mle_params; return gamma.mean(a,loc,scale), gamma.std(a,loc,scale)
        return 0, 1
    def _fit_t(self, data):
        n=len(data);
        if n==0: return
        mean_data,sum_sq_diff=data.mean(),((data-data.mean())**2).sum()
        self.alpha_n=self.alpha_0+n/2; self.beta_n=self.beta_0+0.5*sum_sq_diff+(n*self.nu_0)/(self.nu_0+n)*0.5*(mean_data-self.mu_0)**2
        self.mu_n=(self.nu_0*self.mu_0+n*mean_data)/(self.nu_0+n); self.nu_n=self.nu_0+n
    def _fit_beta(self, data):
        from scipy.stats import beta
        data_norm=data.clip(0.0001,0.9999);
        if len(data_norm)<2: return
        if data_norm.std()<1e-6: data_norm+=np.random.normal(0,1e-5,len(data_norm))
        try: self.mle_params=beta.fit(data_norm,floc=0,fscale=1)
        except Exception: pass
    def _fit_gamma(self, data):
        from scipy.stats import gamma
        if len(data)<2: return
        if data.std()<1e-6: data+=np.random.normal(0,1e-5,len(data))
        try: self.mle_params=gamma.fit(data.clip(0.0001),floc=0)
        except Exception: pass

class TradingEnvironment(gym.Env):
    def __init__(self, df, surprise_scores, initial_balance=100000, trade_fee=0.001, reward_scheme='sortino'):
        super().__init__(); self.df = df.iloc[-len(surprise_scores):].reset_index(drop=True); self.features = surprise_scores
        self.initial_balance = initial_balance; self.trade_fee = trade_fee; self.reward_scheme = reward_scheme
        self.action_space = gym.spaces.Tuple((gym.spaces.Discrete(3), gym.spaces.Box(low=np.array([1.0, 0.0, 0.0]), high=np.array([3.0, 0.1, 0.2]), dtype=np.float32)))
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.features.shape[1],), dtype=np.float32)
        self._start_tick = 0; self.returns_history = []
        self.portfolio_history = deque(maxlen=22)
    def _get_observation(self): return self.features[self._current_step]
    def _get_info(self): return {'portfolio_value': self.portfolio_value}
    def reset(self, seed=None):
        super().reset(seed=seed); self._current_step = self._start_tick
        self.balance, self.shares_held, self.position = self.initial_balance, 0, 0
        self.portfolio_value, self.entry_price, self.stop_loss_price = self.balance, 0, 0
        self.done, self.returns_history = False, [];
        self.portfolio_history.clear()
        self.portfolio_history.append(self.initial_balance)
        return self._get_observation(), self._get_info()
    def step(self, action):
        if self.done: return self.reset()
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
            if self.reward_scheme == 'sortino':
                negative_returns = relevant_returns[relevant_returns < 0]; downside_deviation = np.std(negative_returns) if len(negative_returns) > 1 else 0
                if downside_deviation != 0: reward = np.tanh(mean_return / downside_deviation * np.sqrt(252*7))
                elif mean_return > 0: reward = 1.0
            elif self.reward_scheme == 'sharpe':
                std_dev_returns = np.std(relevant_returns)
                if std_dev_returns != 0: reward = np.tanh(mean_return / std_dev_returns * np.sqrt(252*7))
                elif mean_return > 0: reward = 1.0
            elif self.reward_scheme == 'calmar':
                annualized_return = mean_return * (252 * 7)
                portfolio_values = np.array(self.portfolio_history)
                running_max = np.maximum.accumulate(portfolio_values)
                drawdowns = (running_max - portfolio_values) / running_max
                max_drawdown = np.max(drawdowns)
                if max_drawdown > 0:
                    reward = np.tanh(annualized_return / max_drawdown)
                elif mean_return > 0:
                    reward = 1.0
        if trade_executed: reward -= trade_penalty
        if (self.position==1 and current_price>self.df.iloc[self._current_step]['Open']) or (self.position==-1 and current_price<self.df.iloc[self._current_step]['Open']): reward += trend_reward_bonus
        self._current_step+=1
        if self._current_step >= len(self.features): self.done = True
        return (self.features[self._current_step] if not self.done else self.features[-1]), reward, self.done, False, self._get_info()

class RolloutBuffer:
    def __init__(self): self.actions,self.states,self.logprobs,self.rewards,self.is_terminals,self.values=[],[],[],[],[],[]
    def clear(self): del self.actions[:],self.states[:],self.logprobs[:],self.rewards[:],self.is_terminals[:],self.values[:]

class ActorCritic(nn.Module):
    def __init__(self, n_observations, n_discrete_actions, n_continuous_actions):
        super(ActorCritic, self).__init__()
        self.shared_body=nn.Sequential(nn.Linear(n_observations,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU())
        self.policy_head, self.param_head, self.value_head = nn.Linear(256, n_discrete_actions), nn.Linear(256, n_continuous_actions), nn.Linear(256, 1)
        self.log_std_head=nn.Parameter(torch.zeros(n_continuous_actions))
    def forward(self, x):
        x=self.shared_body(x)
        return torch.softmax(self.policy_head(x),dim=-1), torch.tanh(self.param_head(x)), torch.exp(self.log_std_head), self.value_head(x)

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
            next_non_terminal = 1.0 - is_terminals[t] if t==len(rewards)-1 else 1.0 - is_terminals[t+1]
            next_value = 0 if t==len(rewards)-1 else values[t+1]
            delta = rewards[t] + self.gamma*next_value*next_non_terminal - values[t]
            advantages[t]=last_advantage=delta+self.gamma*self.gae_lambda*next_non_terminal*last_advantage
        returns = advantages + values; advantages = (advantages - advantages.mean())/(advantages.std()+1e-8)
        old_states = torch.FloatTensor(np.array(self.buffer.states))
        old_discrete_actions, old_tanh_actions = torch.tensor([a[0] for a in self.buffer.actions],dtype=torch.int64), torch.tensor(np.array([a[1] for a in self.buffer.actions]),dtype=torch.float32)
        old_logprobs = torch.tensor(self.buffer.logprobs,dtype=torch.float32)
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