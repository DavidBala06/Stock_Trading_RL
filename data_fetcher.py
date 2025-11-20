import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings

# Suppress the UserWarning from the gym library
warnings.filterwarnings("ignore", category=UserWarning)

def get_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Downloads historical stock data using yfinance."""
    print(f"Downloading data for {ticker}...")
    df = yf.download(ticker, start=start_date, end=end_date)
    df.drop(columns=['Adj Close'], inplace=True, errors='ignore')
    return df

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates simple technical indicators (RSI, MACD) and adds them to the DataFrame."""
    # Simple Moving Average (SMA)
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    
    # Relative Strength Index (RSI - simplified calculation)
    # Note: A full RSI calculation is complex, this is a placeholder
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    RS = gain / loss
    df['RSI'] = 100 - (100 / (1 + RS))
    
    # Drop rows with NaN values resulting from rolling window calculations
    df.dropna(inplace=True)
    return df

def preprocess_data(df: pd.DataFrame):
    feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_10', 'RSI']
    df_features = df[feature_columns].copy()
    
    df_features = df_features.astype('float32')
    raw_prices = df['Close'].to_numpy().flatten()
    
    scaler = MinMaxScaler()
    df_features[:] = scaler.fit_transform(df_features)
    
    return df_features.to_numpy(), raw_prices

# Example usage (will be called in train.py)
# if __name__ == '__main__':
#     raw_df = get_stock_data('TSLA', '2020-01-01', '2024-01-01')
#     df_with_indicators = add_technical_indicators(raw_df)
#     data_array = preprocess_data(df_with_indicators)
#     print(f"Final data shape: {data_array.shape}")