import pandas as pd

def calculate_indicators(price_data, short_window=20, long_window=50, std_dev=2):
    """
    Calculates Bollinger Bands and Moving Average Crossovers.
    """
    df = price_data.copy()
    
    # --- Bollinger Bands ---
    df['bb_middle'] = df['close'].rolling(window=short_window).mean()
    df['bb_std'] = df['close'].rolling(window=short_window).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * std_dev)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * std_dev)
    
    # Generate Bollinger Band signals
    df['bb_buy_signal'] = (df['close'] < df['bb_lower'])
    df['bb_sell_signal'] = (df['close'] > df['bb_upper'])
    
    # --- Moving Average Crossover ---
    df['ma_short'] = df['close'].rolling(window=short_window).mean()
    df['ma_long'] = df['close'].rolling(window=long_window).mean()
    
    # Generate Moving Average signals
    df['ma_buy_signal'] = (df['ma_short'].shift(1) < df['ma_long'].shift(1)) & (df['ma_short'] > df['ma_long'])
    df['ma_sell_signal'] = (df['ma_short'].shift(1) > df['ma_long'].shift(1)) & (df['ma_short'] < df['ma_long'])
    
    return df