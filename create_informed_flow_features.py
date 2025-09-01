import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import pandas_ta as ta
import warnings

warnings.filterwarnings("ignore")

def create_smart_money_features(ticker="QQQ", years=7):
    """
    Downloads data and engineers features designed to detect the
    footprints of institutional or "smart money" traders.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years*365)
    
    print(f"Fetching {years} years of data for {ticker}...")
    # --- FIX: Corrected the 'end' parameter in the function call ---
    df = yf.download(ticker, start=start_date, end=end_date, interval='1d')
    if df.empty:
        print("No data fetched. Exiting.")
        return None

    # --- FIX: Flatten the column index from yfinance ---
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # --- END FIX ---

    # --- Feature Engineering ---
    print("Engineering features to detect informed flow...")

    # 1. Volume Spike: How unusual is today's volume?
    # We compare today's volume to its 30-day moving average.
    df['volume_ma_30'] = df['Volume'].rolling(window=30).mean()
    df['volume_spike'] = df['Volume'] / df['volume_ma_30']

    # 2. Price Change: Simple daily log return.
    df['price_change'] = np.log(df['Close'] / df['Close'].shift(1))

    # 3. Price-Volume Interaction: Does high volume confirm the price move?
    # This is a crucial feature. A big price move on a huge volume spike
    # is a strong signal of conviction.
    df['price_vol_interaction'] = df['price_change'] * df['volume_spike']

    # 4. Volatility: Average True Range (ATR) to measure market chop.
    # We use pandas_ta to calculate the 14-day ATR.
    df.ta.atr(length=14, append=True)
    # The column will be named 'ATRr_14' (normalized ATR)

    # --- Finalizing the Feature Set ---
    # These are the variables our model will see.
    feature_names = [
        'volume_spike',
        'price_change',
        'price_vol_interaction',
        'ATRr_14'
    ]

    # Drop any rows with NaN values created by the moving averages/indicators
    df.dropna(inplace=True)

    print("Feature creation complete.")
    print("Features to be used:", feature_names)
    print("Data shape:", df[feature_names].shape)

    return df[feature_names]

if __name__ == '__main__':
    features = create_smart_money_features()
    
    if features is not None:
        # Save the engineered features to a CSV file.
        # This file will be used by our next script to train the model.
        output_path = 'data/informed_flow_features.csv'
        features.to_csv(output_path)
        print(f"\nSuccessfully saved features to '{output_path}'")
        print("\nData Preview:")
        print(features.tail())

