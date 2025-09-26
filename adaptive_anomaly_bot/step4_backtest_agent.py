import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import gymnasium as gym
import os
import matplotlib.pyplot as plt

# --- Configuration ---
DATA_DIR = "final_labeled_data"
TICKER = "QQQ"
MODEL_PATH = "ppo_volatility_agent_v2.pth" # Use the new V2 agent
OUTPUT_FILE = "backtest_performance_v2.png" # Save to a new file

# --- Re-define Environment and Agent Network (needed for loading) ---

class VolatilityBreakoutEnv(gym.Env):
    def __init__(self, features, market_data, initial_balance=100000, trade_fee=0.001):
        super().__init__()
        self.features = features
        self.df = market_data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.trade_fee = trade_fee
        self.action_space = gym.spaces.Discrete(3) 
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(features.shape[1],), dtype=np.float32
        )
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        self._current_step = 1
        self.balance = self.initial_balance
        self.shares_held = 0
        self.entry_price = 0
        self.position = 0
        self.entry_portfolio_value = 0
        self.done = False
        return self.features[self._current_step], {}

    def step(self, action):
        current_price = self.df.iloc[self._current_step]['Close']
        is_breakout = (
            self.df.iloc[self._current_step - 1]['regime'] == 0 and 
            self.df.iloc[self._current_step]['regime'] == 1
        )

        if self.position == 0 and is_breakout:
            if action == 1 or action == 2:
                self.entry_portfolio_value = self._get_portfolio_value()
                self.shares_held = self.balance * (1 - self.trade_fee) / current_price
                self.balance = 0
                self.position = 1 if action == 1 else -1
                self.entry_price = current_price
        elif self.position != 0:
            back_to_compression = self.df.iloc[self._current_step]['regime'] == 0
            agent_exits = action == 0
            if back_to_compression or agent_exits:
                if self.position == 1:
                    self.balance = self.shares_held * current_price * (1 - self.trade_fee)
                elif self.position == -1:
                    profit_loss = (self.entry_price - current_price) * self.shares_held
                    self.balance = self.entry_portfolio_value + profit_loss
                    self.balance *= (1 - self.trade_fee)
                self.shares_held = 0; self.position = 0; self.entry_price = 0

        self._current_step += 1
        if self._current_step >= len(self.features) - 1:
            self.done = True
        
        return self.features[self._current_step], 0, self.done, False, {}

    def _get_portfolio_value(self):
        current_price = self.df.iloc[self._current_step]['Close']
        if self.position == 1:
            return self.shares_held * current_.price
        elif self.position == -1:
            profit_loss = (self.entry_price - current_price) * self.shares_held
            return self.entry_portfolio_value + profit_loss
        else:
            return self.balance

class ActorCritic(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(nn.Linear(n_observations, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, n_actions), nn.Softmax(dim=-1))
        self.critic = nn.Sequential(nn.Linear(n_observations, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 1))
    def forward(self, x):
        return self.actor(x), self.critic(x)

# --- Performance Metrics ---
def calculate_sharpe_ratio(returns, periods_per_year=252*7):
    if returns.std() == 0: return 0.0
    return np.sqrt(periods_per_year) * returns.mean() / returns.std()

def calculate_max_drawdown(equity_curve):
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    return drawdown.min()

# --- Backtesting Execution ---
if __name__ == "__main__":
    print("--- Loading Data and Trained Agent for Backtesting ---")
    
    market_data_path = os.path.join(DATA_DIR, f"{TICKER}_labeled_market_data.csv")
    features_path = os.path.join(DATA_DIR, f"{TICKER}_features_and_labels.npz")
    
    market_df = pd.read_csv(market_data_path, index_col=0, parse_dates=True)
    feature_data = np.load(features_path)
    features = feature_data['features']
    
    if len(features) != len(market_df):
        min_len = min(len(features), len(market_df))
        features = features[:min_len]; market_df = market_df.iloc[:min_len]
        
    policy = ActorCritic(features.shape[1], 3)
    policy.load_state_dict(torch.load(MODEL_PATH))
    policy.eval()

    env = VolatilityBreakoutEnv(features, market_df)
    
    print("--- Running Backtest Simulation ---")
    state, _ = env.reset()
    done = False
    portfolio_history = [env.initial_balance]

    while not done:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            action_probs, _ = policy(state_tensor)
            action = torch.argmax(action_probs).item()
        
        state, _, done, _, _ = env.step(action)
        portfolio_history.append(env._get_portfolio_value())

    print("--- Backtest Complete. Calculating Performance... ---")

    agent_equity = pd.Series(portfolio_history, index=market_df.index[:len(portfolio_history)])
    agent_returns = agent_equity.pct_change().dropna()
    
    buy_hold_equity = env.initial_balance * (market_df['Close'] / market_df['Close'].iloc[0])
    buy_hold_returns = buy_hold_equity.pct_change().dropna()

    agent_total_return = (agent_equity.iloc[-1] / agent_equity.iloc[0] - 1) * 100
    bh_total_return = (buy_hold_equity.iloc[-1] / buy_hold_equity.iloc[0] - 1) * 100
    agent_sharpe = calculate_sharpe_ratio(agent_returns)
    bh_sharpe = calculate_sharpe_ratio(buy_hold_returns)
    agent_mdd = calculate_max_drawdown(agent_equity) * 100
    bh_mdd = calculate_max_drawdown(buy_hold_equity) * 100

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 7))
    
    ax.plot(agent_equity.index, agent_equity / agent_equity.iloc[0], label="Volatility Breakout Agent v2", color='royalblue', lw=2)
    ax.plot(buy_hold_equity.index, buy_hold_equity / buy_hold_equity.iloc[0], label="Buy & Hold Baseline", color='gray', linestyle='--')
    
    ax.set_title(f"Agent Performance vs. Buy & Hold for {TICKER}", fontsize=16)
    ax.set_ylabel("Normalized Portfolio Value", fontsize=12)
    ax.legend(loc='upper left', fontsize=12)
    
    stats_text = (
        f"Agent Performance (v2):\n"
        f"----------------------\n"
        f"Total Return: {agent_total_return:.2f}%\n"
        f"Sharpe Ratio: {agent_sharpe:.2f}\n"
        f"Max Drawdown: {agent_mdd:.2f}%\n\n"
        f"Buy & Hold Performance:\n"
        f"----------------------\n"
        f"Total Return: {bh_total_return:.2f}%\n"
        f"Sharpe Ratio: {bh_sharpe:.2f}\n"
        f"Max Drawdown: {bh_mdd:.2f}%"
    )
    
    ax.text(0.02, 0.4, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
    
    plt.savefig(OUTPUT_FILE)
    print(f"\n✅ Backtest complete. Performance chart saved to {OUTPUT_FILE}")