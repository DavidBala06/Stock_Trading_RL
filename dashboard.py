import streamlit as st
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
import plotly.graph_objects as go
from data_fetcher import get_stock_data, add_technical_indicators, preprocess_data
from trading_env import StockTradingEnv

st.set_page_config(layout="wide")
st.title("AI Stock Trading Bot Dashboard")

# --- Sidebar Controls ---
ticker = st.sidebar.text_input("Stock Ticker", "TSLA")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-01-01"))
initial_balance = st.sidebar.number_input("Initial Balance", value=10000)

if st.sidebar.button("Run Backtest"):
    with st.spinner("Downloading Data and Running Agent..."):
        
        # 1. Prepare Data
        raw_df = get_stock_data(ticker, str(start_date), str(end_date))
        if len(raw_df) < 20:
            st.error("Not enough data. Please choose a longer time range.")
        else:
            df_with_indicators = add_technical_indicators(raw_df)
            data_array, raw_prices = preprocess_data(df_with_indicators)
            
            # 2. Load Model and Env
            # NOTE: Make sure you have a trained model saved as 'ppo_stock_trader_model.zip'
            # You might need to rename your saved file from train.py to this, or update the path below.
            try:
                # Try to find the most recent model file in your folder
                model = PPO.load("ppo_stock_trader_TSLA_100000.zip") 
            except:
                st.error("Model file not found! Please run train.py first.")
                st.stop()

            env = StockTradingEnv(data_array, raw_prices, initial_balance)
            
            # 3. Run Simulation
            obs, _ = env.reset()
            done = False
            
            portfolio_values = []
            actions_history = []
            
            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                portfolio_values.append(env.net_worth)
                actions_history.append(action)

            # 4. Visualize Results
            
            # Create DataFrame for plotting
            result_df = df_with_indicators.iloc[:len(portfolio_values)].copy()
            result_df['Agent_Balance'] = portfolio_values
            
            # Calculate Buy & Hold (Comparison)
            buy_hold_shares = initial_balance / raw_prices[0]
            result_df['Buy_and_Hold'] = buy_hold_shares * result_df['Close']
            
            # --- Metrics Row ---
            total_return = ((env.net_worth - initial_balance) / initial_balance) * 100
            bh_return = ((result_df['Buy_and_Hold'].iloc[-1] - initial_balance) / initial_balance) * 100
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Final Portfolio Value", f"${env.net_worth:,.2f}", f"{total_return:.2f}%")
            col2.metric("Buy & Hold Value", f"${result_df['Buy_and_Hold'].iloc[-1]:,.2f}", f"{bh_return:.2f}%")
            col3.metric("Total Trades", len([x for x in actions_history if x != 0]))

            # --- Interactive Plot ---
            st.subheader("Agent Performance vs Market")
            fig = go.Figure()
            
            # Agent Line
            fig.add_trace(go.Scatter(
                x=result_df.index, y=result_df['Agent_Balance'],
                mode='lines', name='AI Agent', line=dict(color='green', width=2)
            ))
            
            # Buy & Hold Line
            fig.add_trace(go.Scatter(
                x=result_df.index, y=result_df['Buy_and_Hold'],
                mode='lines', name='Buy & Hold', line=dict(color='gray', dash='dash')
            ))
            
            st.plotly_chart(fig, use_container_width=True)