import numpy as np
import warnings
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from data_fetcher import get_stock_data, add_technical_indicators, preprocess_data
from trading_env import StockTradingEnv

# Suppress the UserWarning from the gym library
warnings.filterwarnings("ignore", category=UserWarning)

# --- CONFIGURATION ---
TICKER = 'TSLA'
START_DATE = '2020-01-01'
END_DATE = '2024-01-01'
TOTAL_TIMESTEPS = 100000 
INITIAL_BALANCE = 10000

def run_training():
    """Fetches data, creates environment, and trains the PPO agent."""
    
    # 1. Data Pipeline
    raw_df = get_stock_data(TICKER, START_DATE, END_DATE)
    df_with_indicators = add_technical_indicators(raw_df)
    data_array, raw_prices = preprocess_data(df_with_indicators)
    if hasattr(data_array, "shape"):
        print(f"Preprocessed data shape (timesteps, features): {data_array.shape}")
    else:
        print(f"Preprocessed data type: {type(data_array)}, length: {len(data_array)}")
    
    # 2. Environment Creation
    
    env_kwargs = {
        'data_array': data_array, 
        'raw_prices': raw_prices, # Pass raw prices to env
        'initial_balance': INITIAL_BALANCE
    }
    
    # Stable-Baselines3 algorithms require vectorized environments for parallelization and consistent API, 
    env = make_vec_env(StockTradingEnv, n_envs=1, env_kwargs=env_kwargs, seed=42)
    
    # 3. Model Definition and Training
    # PPO (Proximal Policy Optimization) is state-of-the-art for RL
    model = PPO(
        "MlpPolicy", # MlpPolicy: Uses a simple Multi-Layer Perceptron (standard for non-image data)
        env, 
        verbose=1, 
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        tensorboard_log="./ppo_trading_log/"
    )
    
    print(f"\nStarting training for {TOTAL_TIMESTEPS} timesteps...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    
    # 4. Save the Model
    model_path = f"ppo_stock_trader_{TICKER}_{TOTAL_TIMESTEPS}.zip"
    model.save(model_path)
    print(f"\nTraining complete. Model saved to: {model_path}")

    # 5. Test the Model (Optional)
    print("\n--- Testing the Trained Agent (One Episode) ---")
    obs = env.reset()
    done = False
    
    # We need to access the actual environment inside the vector wrapper to render it
    test_env = env.envs[0] 
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        
        # 'dones' is an array because it's a vectorized env, we just check the first one
        done = dones[0] 
        
        test_env.render()

if __name__ == '__main__':
    run_training()