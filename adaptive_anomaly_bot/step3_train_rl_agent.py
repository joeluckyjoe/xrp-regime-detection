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

# --- Configuration ---
DATA_DIR = "final_labeled_data"
TICKER = "QQQ"
MODEL_SAVE_PATH = "ppo_volatility_agent.pth"
TRAINING_EPISODES = 50
UPDATE_TIMESTEP = 2000

# --- Trading Environment for Volatility Breakouts ---
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
        self.position = 0 # 0=FLAT, 1=LONG, -1=SHORT
        self.done = False
        return self.features[self._current_step], {}

    def step(self, action):
        current_price = self.df.iloc[self._current_step]['Close']
        old_portfolio_value = self._get_portfolio_value()

        # Core Breakout Logic
        is_breakout = (
            self.df.iloc[self._current_step - 1]['regime'] == 0 and 
            self.df.iloc[self._current_step]['regime'] == 1
        )

        if self.position == 0 and is_breakout: # Can only enter on a breakout signal
            if action == 1: # Go Long
                self.shares_held = self.balance * (1 - self.trade_fee) / current_price
                self.balance = 0
                self.position = 1
                self.entry_price = current_price
            elif action == 2: # Go Short
                self.shares_held = self.balance * (1 - self.trade_fee) / current_price
                self.balance = 0
                self.position = -1
                self.entry_price = current_price
        elif self.position != 0: # Can exit at any time
            back_to_compression = self.df.iloc[self._current_step]['regime'] == 0
            agent_exits = action == 0

            if back_to_compression or agent_exits:
                if self.position == 1: # Close Long
                    self.balance = self.shares_held * current_price * (1 - self.trade_fee)
                elif self.position == -1: # Close Short
                    profit_loss = (self.entry_price - current_price) * self.shares_held
                    self.balance = (self.entry_price * self.shares_held) + profit_loss
                    self.balance *= (1 - self.trade_fee)

                self.shares_held = 0
                self.position = 0
                self.entry_price = 0

        self._current_step += 1
        if self._current_step >= len(self.features) - 1:
            self.done = True

        new_portfolio_value = self._get_portfolio_value()
        reward = (new_portfolio_value - old_portfolio_value) / old_portfolio_value if old_portfolio_value != 0 else 0
        
        return self.features[self._current_step], reward, self.done, False, {}

    def _get_portfolio_value(self):
        current_price = self.df.iloc[self._current_step]['Close']
        if self.position == 1:
            return self.shares_held * current_price
        elif self.position == -1:
            profit_loss = (self.entry_price - current_price) * self.shares_held
            return (self.entry_price * self.shares_held) + profit_loss
        else:
            return self.balance

# --- PPO Agent Code ---
class RolloutBuffer:
    # --- FIX: Corrected the syntax for creating empty lists ---
    def __init__(self):
        self.actions, self.states, self.logprobs, self.rewards, self.is_terminals, self.values = [], [], [], [], [], []
    def clear(self):
        del self.actions[:]; del self.states[:]; del self.logprobs[:]; del self.rewards[:]; del self.is_terminals[:]; del self.values[:]

class ActorCritic(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(nn.Linear(n_observations, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, n_actions), nn.Softmax(dim=-1))
        self.critic = nn.Sequential(nn.Linear(n_observations, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 1))
    def forward(self, x):
        return self.actor(x), self.critic(x)

class PPOAgent:
    def __init__(self, env, lr=3e-4, gamma=0.99, K_epochs=4, eps_clip=0.2):
        self.env=env; self.gamma=gamma; self.eps_clip=eps_clip; self.K_epochs=K_epochs
        self.policy=ActorCritic(env.observation_space.shape[0],env.action_space.n)
        self.optimizer=optim.Adam(self.policy.parameters(),lr=lr)
        self.policy_old=ActorCritic(env.observation_space.shape[0],env.action_space.n)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss=nn.MSELoss(); self.buffer=RolloutBuffer()

    def select_action(self, state):
        with torch.no_grad():
            state_tensor=torch.FloatTensor(state)
            action_probs,state_val=self.policy_old(state_tensor)
            dist=Categorical(action_probs); action=dist.sample()
            return action.item(),dist.log_prob(action).item(),state_val.item()

    def learn(self):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal: discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = torch.squeeze(torch.tensor(np.array(self.buffer.states), dtype=torch.float32))
        old_actions = torch.squeeze(torch.tensor(self.buffer.actions, dtype=torch.long))
        old_logprobs = torch.squeeze(torch.tensor(self.buffer.logprobs, dtype=torch.float32))

        for _ in range(self.K_epochs):
            action_probs, state_values = self.policy(old_states)
            dist = Categorical(action_probs)
            logprobs = dist.log_prob(old_actions)
            dist_entropy = dist.entropy()
            
            state_values = torch.squeeze(state_values)
            
            advantages = rewards - state_values.detach()
            ratios = torch.exp(logprobs - old_logprobs)
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

# --- Main Training Execution ---
if __name__ == "__main__":
    print("--- Loading Labeled Data for Layer 3 ---")
    market_data_path = os.path.join(DATA_DIR, f"{TICKER}_labeled_market_data.csv")
    features_path = os.path.join(DATA_DIR, f"{TICKER}_features_and_labels.npz")
    
    market_df = pd.read_csv(market_data_path, index_col=0, parse_dates=True)
    feature_data = np.load(features_path)
    features = feature_data['features']; labels = feature_data['labels']
    
    if len(features) != len(market_df):
        min_len = min(len(features), len(market_df))
        features = features[:min_len]; market_df = market_df.iloc[:min_len]
        print(f"Data aligned to length: {min_len}")
        
    print("--- Initializing Agent and Environment ---")
    env = VolatilityBreakoutEnv(features, market_df)
    agent = PPOAgent(env)
    
    print(f"--- Starting Training for {TRAINING_EPISODES} Episodes ---")
    time_step = 0
    
    for episode in tqdm(range(1, TRAINING_EPISODES + 1)):
        state, _ = env.reset()
        done = False
        while not done:
            time_step += 1
            action, log_prob, state_val = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            
            agent.buffer.states.append(state); agent.buffer.actions.append(action)
            agent.buffer.logprobs.append(log_prob); agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(done); agent.buffer.values.append(state_val)

            state = next_state
            
            if time_step % UPDATE_TIMESTEP == 0:
                agent.learn()
                time_step = 0

    print("--- Training Complete ---")
    
    torch.save(agent.policy.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ Trained agent saved to {MODEL_SAVE_PATH}")