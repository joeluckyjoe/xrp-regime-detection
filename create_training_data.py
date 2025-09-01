import pandas as pd
import numpy as np
import warnings
import pandas_ta as ta
import yfinance as yf
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

def fetch_yfinance_data(ticker, start_date, end_date):
    """Fetches historical daily data from Yahoo Finance."""
    df = yf.download(ticker, start=start_date, end=end_date, interval='1d')
    df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
    df = df.tz_localize('UTC')
    return df

def create_features_and_target(data, short_window=20, long_window=50):
    """Creates features from indicators and a target for our model."""
    df = data.copy()

    # --- THIS IS THE FIX ---
    # Calculate all standard indicators first
    df['ma_200'] = df['close'].rolling(window=200).mean()
    df.ta.rsi(length=14, append=True)
    
    bbands = df.ta.bbands(length=short_window)
    # Find the correct column names for the lower and upper bands
    lower_band_col = [col for col in bbands.columns if 'BBL' in col][0]
    upper_band_col = [col for col in bbands.columns if 'BBU' in col][0]
    df = df.join(bbands[[lower_band_col, upper_band_col]])
    
    df['ma_short'] = df['close'].rolling(window=short_window).mean()
    df['ma_long'] = df['close'].rolling(window=long_window).mean()
    
    # Now, add our custom feature columns
    df['price_vs_ma200'] = (df['close'] - df['ma_200']) / df['ma_200']
    df['price_vs_bb'] = (df['close'] - df[lower_band_col]) / (df[upper_band_col] - df[lower_band_col])
    df['ma_crossover'] = (df['ma_short'] - df['ma_long']) / df['ma_long']

    # --- Create Target ---
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    # Select final columns and drop NaNs
    feature_names = ['price_vs_ma200', 'RSI_14', 'price_vs_bb', 'ma_crossover']
    df = df[feature_names + ['target']].dropna()
    
    return df[feature_names], df['target']

if __name__ == '__main__':
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    start_str, end_str = start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
    
    print("Fetching 5 years of NASDAQ (QQQ) data...")
    market_data = fetch_yfinance_data("QQQ", start_str, end_str)

    print("Creating features and target...")
    features, target = create_features_and_target(market_data)
    
    features.to_csv('data/nasdaq_features.csv')
    target.to_csv('data/nasdaq_target.csv', header=['target'])

    print(f"\nTraining data created successfully!")
    print(f"Saved {len(features)} data points.")
    print("\nFeatures Preview:")
    print(features.tail())