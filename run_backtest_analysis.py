import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pygam import LinearGAM, s
from datetime import datetime, timedelta
import ccxt
import warnings

# Suppress the known pygam warning for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

def fetch_recent_data(hours_back=120, timeframe='5m', exchange_name='binance'):
    """Fetches recent historical OHLCV data."""
    exchange_class = getattr(ccxt, exchange_name)
    exchange = exchange_class()
    since_datetime = datetime.utcnow() - timedelta(hours=hours_back)
    since_timestamp = int(since_datetime.timestamp() * 1000)
    ohlcv = exchange.fetch_ohlcv('XRP/USDT', timeframe, since=since_timestamp, limit=2000)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def generate_historical_signals(data, short_window=20, long_window=50):
    """
    Analyzes the volume cycle and generates historical signals for the entire dataset.
    """
    # 1. Prepare the volume data
    log_volume = np.log(data['volume']).dropna()
    scaler = StandardScaler()
    scaled_log_volume = scaler.fit_transform(log_volume.values.reshape(-1, 1))
    time_idx = np.arange(len(scaled_log_volume))
    time_of_day = time_idx % 288

    # 2. Fit the GAM to learn the cycle
    gam = LinearGAM(s(0, n_splines=15, basis='cp')).fit(time_of_day, scaled_log_volume)
    
    # 3. Generate predictions for the entire history
    historical_predictions = gam.predict(time_of_day)
    cycle_mean = historical_predictions.mean()
    
    # 4. Create a regime column
    regime = np.where(historical_predictions > cycle_mean, 'high_volatility', 'low_volatility')
    
    # Create a new DataFrame for the results
    results_df = data.loc[log_volume.index].copy()
    results_df['regime'] = regime

    # 5. Calculate both types of indicators
    # Bollinger Bands
    results_df['bb_middle'] = results_df['close'].rolling(window=short_window).mean()
    results_df['bb_std'] = results_df['close'].rolling(window=short_window).std()
    results_df['bb_upper'] = results_df['bb_middle'] + (results_df['bb_std'] * 2)
    results_df['bb_lower'] = results_df['bb_middle'] - (results_df['bb_std'] * 2)
    results_df['bb_buy'] = (results_df['close'] < results_df['bb_lower'])
    results_df['bb_sell'] = (results_df['close'] > results_df['bb_upper'])
    
    # Moving Average Crossover
    results_df['ma_short'] = results_df['close'].rolling(window=short_window).mean()
    results_df['ma_long'] = results_df['close'].rolling(window=long_window).mean()
    results_df['ma_buy'] = (results_df['ma_short'].shift(1) < results_df['ma_long'].shift(1)) & (results_df['ma_short'] > results_df['ma_long'])
    results_df['ma_sell'] = (results_df['ma_short'].shift(1) > results_df['ma_long'].shift(1)) & (results_df['ma_short'] < results_df['ma_long'])
    
    return results_df

if __name__ == '__main__':
    print("Fetching latest 5-day market data for backtest...")
    market_data = fetch_recent_data(hours_back=120, timeframe='5m')
    
    print("\nGenerating historical signals based on volume cycle...")
    backtest_df = generate_historical_signals(market_data)
    
    # Save results to a file for the notebook to use
    backtest_df.to_csv('data/backtest_results.csv')
    
    print("\n" + "="*50)
    print("Backtest analysis complete.")
    print("Results saved to 'data/backtest_results.csv'")
    print("You can now visualize these results in your notebook.")
    print("="*50)