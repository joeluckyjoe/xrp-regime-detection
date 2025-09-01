import pandas as pd
import numpy as np
import warnings
import yfinance as yf
from datetime import datetime, timedelta
import pandas_ta as ta

warnings.filterwarnings("ignore")

def fetch_yfinance_data(ticker, start_date, end_date):
    """Fetches historical daily data from Yahoo Finance."""
    df = yf.download(ticker, start=start_date, end=end_date, interval='1d')
    if df.empty: return df
    
    # --- THIS IS THE FIX ---
    # Flatten the multi-level column index
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
    df = df.tz_localize('UTC')
    return df

def create_features_and_target(data, short_window=20, long_window=50):
    """Creates features from indicators and a target for our model."""
    df = data.copy()
    
    df['ma_200'] = df['close'].rolling(window=200).mean()
    df.ta.rsi(length=14, append=True)
    df['ma_short'] = df['close'].rolling(window=short_window).mean()
    df['ma_long'] = df['close'].rolling(window=long_window).mean()
    
    df['bb_middle'] = df['close'].rolling(window=short_window).mean()
    df['bb_std'] = df['close'].rolling(window=short_window).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)

    df['price_vs_ma200'] = (df['close'] - df['ma_200']) / df['ma_200']
    df['price_vs_bb'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    df['ma_crossover'] = (df['ma_short'] - df['ma_long']) / df['ma_long']

    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    feature_names = ['price_vs_ma200', 'RSI_14', 'price_vs_bb', 'ma_crossover']
    final_df = df[feature_names + ['target']].dropna()
    
    return final_df[feature_names], final_df['target']

if __name__ == '__main__':
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    start_str, end_str = start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
    
    print("Fetching 5 years of NASDAQ (QQQ) data...")
    market_data = fetch_yfinance_data("QQQ", start_str, end_str)

    if not market_data.empty:
        print("Creating features and target...")
        features, target = create_features_and_target(market_data)
        
        features.to_csv('data/nasdaq_features.csv')
        target.to_csv('data/nasdaq_target.csv', header=['target'])

        print(f"\nTraining data created successfully!")
        print(f"Saved {len(features)} data points.")
        print("\nFeatures Preview:")
        print(features.tail())
    else:
        print("Could not fetch data.")