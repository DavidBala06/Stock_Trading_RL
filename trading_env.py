import numpy as np
import gym
from gym import spaces

# Constants
MAX_ACCOUNT_BALANCE = 100000000000.0
MAX_NUM_SHARES = 2147483647
TRANSACTION_FEE_PCT = 0.001

class StockTradingEnv(gym.Env):
    """
    A custom Gym environment for single-stock trading simulation.
    """
    metadata = {'render_modes': ['human']}
    
    def __init__(self, data_array: np.ndarray, raw_prices: np.ndarray, initial_balance=10000):
        super(StockTradingEnv, self).__init__()
        
        self.stock_data = data_array
        self.raw_prices = raw_prices
        self.initial_balance = initial_balance
        self.max_steps = len(self.stock_data) - 1
        self.current_step = 0

        # --- Portfolio State Initialization ---
        self.cash_balance = initial_balance
        self.shares_held = 0
        self.net_worth = initial_balance
        self.max_net_worth = initial_balance
        
        # Action Space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3) 

        # Observation Space: Market Features + Portfolio Status
        num_market_features = self.stock_data.shape[1] 
        num_portfolio_features = 3 
        state_shape = num_market_features + num_portfolio_features
        
        self.observation_space = spaces.Box(
            low=0, 
            high=1, 
            shape=(state_shape,), 
            dtype=np.float32
        )
        
    def _get_obs(self):
        # 1. Market Features
        # Use flatten() to ensure it's a simple 1D array
        market_features = self.stock_data[self.current_step].flatten()
        
        # 2. Portfolio Features
        # WRAP IN float() to ensure they are single numbers, not arrays
        norm_cash = float(self.cash_balance / MAX_ACCOUNT_BALANCE)
        norm_shares = float(self.shares_held / MAX_NUM_SHARES)
        norm_net_worth = float(self.net_worth / MAX_ACCOUNT_BALANCE)
        
        portfolio_features = np.array([norm_cash, norm_shares, norm_net_worth], dtype=np.float32)
        
        return np.concatenate([market_features, portfolio_features])

    def _take_action(self, action, current_price):
        action_type = ["HOLD", "BUY", "SELL"][action]
        
        if action == 1: # BUY
            shares_to_buy = int(self.cash_balance / current_price)
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price
                fee = cost * TRANSACTION_FEE_PCT
                self.cash_balance -= (cost + fee)
                self.shares_held += shares_to_buy
                # print(f"BOUGHT {shares_to_buy} shares at ${current_price:.2f}")
            else:
                
                # print(f"WANTED TO BUY, BUT NO CASH (Cash: ${self.cash_balance:.2f}, Price: ${current_price:.2f})")
                pass
                
        elif action == 2: # SELL
            if self.shares_held > 0:
                sales_value = self.shares_held * current_price
                fee = sales_value * TRANSACTION_FEE_PCT
                self.cash_balance += (sales_value - fee)
                self.shares_held = 0
                # print(f"SOLD ALL shares at ${current_price:.2f}")
            else:
                 # print(f"WANTED TO SELL, BUT NO SHARES")
                 pass

    def step(self, action):
        prev_net_worth = self.net_worth 
        
        # FORCE FLOAT here to prevent array contamination
        current_raw_price = float(self.raw_prices[self.current_step])
        self._take_action(action, current_raw_price)
        
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        # Calculate new Net Worth
        next_raw_price = float(self.raw_prices[self.current_step]) if not done else current_raw_price
        
        self.net_worth = self.cash_balance + (self.shares_held * next_raw_price)
        self.max_net_worth = max(self.net_worth, self.max_net_worth)
        
        # Calculate Reward
        if prev_net_worth == 0:
             reward = 0
        else:
             reward = (self.net_worth - prev_net_worth) / prev_net_worth
        
        # Penalty for bankruptcy
        if self.net_worth <= 0:
            done = True
            reward = -100

        obs = self._get_obs()
        
        return obs, reward, done, False, {}
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.cash_balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.current_step = 0
        
        return self._get_obs(), {}

    def render(self, mode='human'):
        profit = self.net_worth - self.initial_balance
        print(f'Step: {self.current_step}')
        print(f'Balance: ${self.cash_balance:,.2f} | Shares: {self.shares_held}')
        print(f'Net Worth: ${self.net_worth:,.2f} | Profit: ${profit:,.2f}')
        print(f'--------------------------------')