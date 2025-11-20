Deep Reinforcement Learning Stock Trading Bot:

An autonomous stock trading agent trained using Deep Reinforcement Learning (PPO). This project simulates a trading environment where an AI agent learns to buy, sell, or hold stocks to maximize portfolio returns. 
In backtesting tests on unseen data, the agent successfully identified market downturns and protected capital, outperforming the standard "Buy and Hold" strategy.

Project Overview:

The goal of this project was to create a trading bot that does not rely on hard-coded rules but instead learns a policy through trial and error. 
The system uses the Proximal Policy Optimization (PPO) algorithm to interact with a custom OpenAI Gym environment.
The final result is verified using a custom web dashboard that visualizes the agent's trades and equity curve against the underlying asset.

Key Features:
Custom Gym Environment: A built-from-scratch environment simulating the stock market, accounting for transaction fees and liquidity constraints.
Deep Learning Agent: Utilizes the PPO algorithm from Stable-Baselines3.
Interactive Dashboard: A web-based interface built with Streamlit and Plotly for visualizing backtest results.
Real-Time Data: Fetches historical financial data dynamically using the yfinance library.

Project Structure:
data_fetcher.py: Handles data downloading, cleaning, and feature normalization.
trading_env.py: The custom Gym environment defining the state space, action space, and reward function.
train.py: The main script used to train the agent and save the model.
dashboard.py: The Streamlit application for visualizing the trained agent's performance.
requirements.txt: List of Python dependencies.

Installation and Setup:
1. Clone the repository:
git clone https://github.com/DavidBala06/stock-trading-rl.git
cd stock-trading-rl

2. Create and activate a virtual environment:
python -m venv venv
# Windows:
.\venv\Scripts\Activate.ps1
# Mac/Linux:
source venv/bin/activate

3. Install dependencies:
pip install -r requirements.txt

Usage:
1. Training the Agent:
To train a new model, run the training script. This will download historical data, train the PPO agent for the specified number of timesteps, and save the model as a .zip file.
python train.py
2. Visualizing Results:
Once a model is trained, launch the dashboard to view the performance.
streamlit run dashboard.py

Methodology:
Observation Space (State):
The agent receives a normalized state vector containing:
Market Data: Open, High, Low, Close, Volume.
Technical Indicators: Simple Moving Average (SMA), Relative Strength Index (RSI).
Portfolio State: Current Cash Balance, Shares Held, Total Net Worth.

Action Space:
The agent operates in a discrete action space: 0. Hold: Do nothing.
Buy: Convert available cash into shares.
Sell: Convert all shares into cash.

Reward Function:
The reward is calculated based on the percentage change in the portfolio's Net Worth between time steps. This incentivizes the agent to maximize capital appreciation while avoiding significant drawdowns.

Results:
In the latest backtest on Tesla (TSLA) stock data (2023-2024), the agent achieved:
AI Profit: +81.74%
Market Profit (Buy & Hold): +76.13%
The agent demonstrated an ability to exit the market during high volatility, preserving gains when the underlying asset price dropped.

License:
This project is open-source and available under the MIT License.
