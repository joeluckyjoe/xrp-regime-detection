import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
from collections import namedtuple, deque
from scipy.stats import t, beta, gamma
import yfinance as yf
import pandas_ta as ta

class BayesianAnomalyDetector:
    def __init__(self, distribution_type='t'):
        self.dist_type = distribution_type
        self.alpha_0, self.beta_0, self.mu_0, self.nu_0 = 1.0, 1.0, 0.0, 1.0
        self.reset_posterior()

    def reset_posterior(self):
        self.alpha_n, self.beta_n, self.mu_n, self.nu_n = self.alpha_0, self.beta_0, self.mu_0, self.nu_0
        self.mle_params = None

    def fit(self, data):
        if self.dist_type == 't': self._fit_t(data)
        elif self.dist_type == 'beta': self._fit_beta(data)
        elif self.dist_type == 'gamma': self._fit_gamma(data)

    def compute_surprise(self, x):
        if self.mle_params is None and self.dist_type in ['beta', 'gamma']: return 0.5

        if self.dist_type == 't':
            df, loc, scale = 2 * self.alpha_n, self.mu_n, np.sqrt(self.beta_n / (self.alpha_n * self.nu_n))
            if scale <= 0: return 0.5
            cdf_val, sf_val = t.cdf(x, df=df, loc=loc, scale=scale), t.sf(x, df=df, loc=loc, scale=scale)
        elif self.dist_type == 'beta':
            x_norm = max(0.0001, min(x, 0.9999))
            a, b, loc, scale = self.mle_params
            cdf_val, sf_val = beta.cdf(x_norm, a, b, loc, scale), beta.sf(x_norm, a, b, loc, scale)
        elif self.dist_type == 'gamma':
            a, loc, scale = self.mle_params
            if scale <= 0: return 0.5
            cdf_val, sf_val = gamma.cdf(x, a, loc=loc, scale=scale), gamma.sf(x, a, loc=loc, scale=scale)

        return 1.0 - (2 * min(cdf_val, sf_val))

    def get_distribution_params(self):
        if self.dist_type == 't':
            return self.mu_n, np.sqrt(self.beta_n / (self.alpha_n * self.nu_n))
        elif self.dist_type == 'beta' and self.mle_params:
            a, b, loc, scale = self.mle_params
            return beta.mean(a, b, loc, scale), beta.std(a, b, loc, scale)
        elif self.dist_type == 'gamma' and self.mle_params:
            a, loc, scale = self.mle_params
            return gamma.mean(a, loc, scale), gamma.std(a, loc, scale)
        return 0, 1

    def _fit_t(self, data):
        n = len(data)
        if n == 0: return
        mean_data, sum_sq_diff = data.mean(), ((data - data.mean()) ** 2).sum()
        self.alpha_n = self.alpha_0 + n / 2
        self.beta_n = self.beta_0 + 0.5 * sum_sq_diff + (n * self.nu_0) / (self.nu_0 + n) * 0.5 * (mean_data - self.mu_0)**2
        self.mu_n = (self.nu_0 * self.mu_0 + n * mean_data) / (self.nu_0 + n)
        self.nu_n = self.nu_0 + n

    def _fit_beta(self, data):
        data_norm = data.clip(0.0001, 0.9999)
        if len(data_norm) < 2: return
        try: self.mle_params = beta.fit(data_norm, floc=0, fscale=1)
        except Exception: pass

    def _fit_gamma(self, data):
        if len(data) < 2: return
        try: self.mle_params = gamma.fit(data.clip(0.0001), floc=0)
        except Exception: pass

def fetch_and_prepare_data(ticker="QQQ", period=None, interval="1h", start=None, end=None):
    print("--- Using robust merge_asof strategy with UTC conversion. ---")

    print(f"Fetching {interval} data for {ticker}...")
    ticker_data = yf.Ticker(ticker).history(period=period, interval=interval, start=start, end=end)
    if ticker_data.empty:
        print(f"No data found for {ticker}")
        return None

    print(f"Fetching DAILY data for VIX...")
    vix_data = yf.Ticker("^VIX").history(period=period, interval="1d", start=start, end=end)
    if vix_data.empty:
        print("Warning: Could not fetch VIX data.")
        return None

    ticker_data.index = ticker_data.index.tz_convert('UTC')
    vix_data.index = vix_data.index.tz_convert('UTC')

    vix_data.rename(columns={'Close': 'vix'}, inplace=True)

    data = pd.merge_asof(
        left=ticker_data.sort_index(),
        right=vix_data[['vix']].sort_index(),
        left_index=True,
        right_index=True,
        direction='backward'
    )

    print("Data fetched and aligned. Calculating indicators...")
    data['returns'] = (data['Close'] - data['Open']) / data['Open']
    data.ta.rsi(length=14, append=True)
    data.ta.atr(length=14, append=True)

    sma_long = data.ta.sma(close=data['Close'], length=200)
    data['sma_dist'] = (data['Close'] - sma_long) / sma_long

    data.rename(columns={"RSI_14": "rsi", "ATRr_14": "atr"}, inplace=True)
    data['rsi'] = data['rsi'] / 100.0
    atr_long = data.ta.sma(close=data['atr'], length=200)
    data['vol_regime'] = data['atr'] / atr_long

    data.dropna(inplace=True)

    if data.empty:
        print("CRITICAL ERROR: Dataframe is empty after indicator calculation and dropna().")
        return None

    print(f"Data preparation complete. Final shape: {data.shape}")
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'returns', 'rsi', 'atr', 'sma_dist', 'vol_regime', 'vix']
    return data[required_cols]

class TradingEnvironment(gym.Env):
    def __init__(self, df, surprise_scores, initial_balance=100000, trade_fee=0.001, stop_loss_atr_multiplier=2.0, trade_penalty=0.1, trend_reward_bonus=0.01):
        super().__init__()
        self.df = df.iloc[-len(surprise_scores):].reset_index(drop=True)
        self.features = surprise_scores
        self.initial_balance, self.trade_fee = initial_balance, trade_fee
        self.stop_loss_atr_multiplier = stop_loss_atr_multiplier
        self.trade_penalty = trade_penalty
        self.trend_reward_bonus = trend_reward_bonus
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(28,), dtype=np.float32)
        self._start_tick = 0
        self.returns_history = []

    def _get_observation(self):
        return self.features[self._current_step]

    def _get_info(self):
        return {'portfolio_value': self.portfolio_value}

    def reset(self, seed=None):
        super().reset(seed=seed)
        self._current_step = self._start_tick
        self.balance, self.shares_held, self.position = self.initial_balance, 0, 0
        self.portfolio_value, self.entry_price, self.stop_loss_price = self.balance, 0, 0
        self.done, self.returns_history = False, []
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.done: return self.reset()
        current_price = self.df.iloc[self._current_step]['Close']
        current_atr = self.df.iloc[self._current_step]['atr']
        old_portfolio_value = self.portfolio_value
        trade_executed = False

        if self.position == 1 and current_price < self.stop_loss_price: action = 2
        elif self.position == -1 and current_price > self.stop_loss_price: action = 1

        if action == 1:
            if self.position == -1:
                self.balance = self.portfolio_value * (1 - self.trade_fee)
                self.position, self.shares_held, self.entry_price = 0, 0, 0
                trade_executed = True
            elif self.position == 0:
                self.position, self.entry_price = 1, current_price
                self.stop_loss_price = current_price - (current_atr * self.stop_loss_atr_multiplier)
                self.shares_held = self.balance / current_price * (1 - self.trade_fee)
                trade_executed = True
        elif action == 2:
            if self.position == 1:
                self.balance = self.shares_held * current_price * (1 - self.trade_fee)
                self.position, self.shares_held, self.entry_price = 0, 0, 0
                trade_executed = True
            elif self.position == 0:
                self.position, self.entry_price = -1, current_price
                self.stop_loss_price = current_price + (current_atr * self.stop_loss_atr_multiplier)
                self.shares_held = self.balance / current_price * (1 - self.trade_fee)
                trade_executed = True

        if self.position == 1: self.portfolio_value = self.shares_held * current_price
        elif self.position == -1: self.portfolio_value = self.balance + (self.entry_price - current_price) * self.shares_held
        else: self.portfolio_value = self.balance

        hourly_return = (self.portfolio_value / old_portfolio_value) - 1 if old_portfolio_value != 0 else 0
        self.returns_history.append(hourly_return)
        reward = 0
        if len(self.returns_history) > 21:
            relevant_returns = np.array(self.returns_history[-21:])
            mean_return = np.mean(relevant_returns)
            negative_returns = relevant_returns[relevant_returns < 0]
            downside_deviation = np.std(negative_returns) if len(negative_returns) > 1 else 0
            if downside_deviation != 0:
                # Original hourly annualization factor
                sortino_ratio = mean_return / downside_deviation * np.sqrt(252 * 7)
                reward = np.tanh(sortino_ratio)
            elif mean_return > 0: reward = 1.0
        if trade_executed: reward -= self.trade_penalty
        current_sma_dist = self.df.iloc[self._current_step]['sma_dist']
        if (self.position == 1 and current_sma_dist > 0) or (self.position == -1 and current_sma_dist < 0):
            reward += self.trend_reward_bonus
        self._current_step += 1
        if self._current_step >= len(self.features): self.done = True
        obs = self.features[self._current_step] if not self.done else self.features[-1]
        return obs, reward, self.done, False, self._get_info()

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
class ReplayMemory:
    def __init__(self, capacity): self.memory = deque([], maxlen=capacity)
    def push(self, *args): self.memory.append(Transition(*args))
    def sample(self, batch_size): return random.sample(self.memory, batch_size)
    def __len__(self): return len(self.memory)

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
    def forward(self, x):
        x = nn.functional.relu(self.layer1(x))
        x = nn.functional.relu(self.layer2(x))
        return self.layer3(x)

class Agent:
    def __init__(self, env):
        self.env = env
        self.BATCH_SIZE, self.GAMMA, self.TAU, self.LR = 128, 0.99, 0.005, 1e-4
        self.EPS_START, self.EPS_END, self.EPS_DECAY = 0.9, 0.05, 1000
        state, _ = self.env.reset()
        n_observations, n_actions = len(state), self.env.action_space.n
        self.policy_net = DQN(n_observations, n_actions)
        self.target_net = DQN(n_observations, n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.memory = ReplayMemory(10000)
        self.steps_done = 0
    def select_action(self, state):
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if random.random() > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], dtype=torch.long)
    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE: return
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(s is not None for s in batch.next_state), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.BATCH_SIZE)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch
        loss = nn.SmoothL1Loss()(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()