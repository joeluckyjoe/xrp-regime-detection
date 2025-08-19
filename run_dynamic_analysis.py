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

def get_dynamic_signal(data):
    """
    Analyzes the volume cycle and returns the current market state.
    """
    # 1. Prepare the data
    log_volume = np.log(data['volume']).dropna()
    scaler = StandardScaler()
    scaled_log_volume = scaler.fit_transform(log_volume.values.reshape(-1, 1))
    
    time_idx = np.arange(len(scaled_log_volume))
    time_of_day = time_idx % 288  # 288 is the number of 5-min intervals in a day

    # 2. Fit the GAM to learn the cycle
    gam = LinearGAM(s(0, n_splines=15, basis='cp')).fit(time_of_day, scaled_log_volume)
    
    # 3. Get the prediction for the most recent data point
    current_cycle_prediction = gam.predict(time_of_day[-1:])
    
    # 4. Get the overall mean of the cycle to use as a threshold
    cycle_mean = gam.predict(time_of_day).mean()
    
    # 5. Make the decision
    print(f"Current Predicted Cycle Value: {current_cycle_prediction[0]:.2f}")
    print(f"Average Cycle Value: {cycle_mean:.2f}")

    if current_cycle_prediction[0] > cycle_mean:
        return "High-Volume / Momentum Phase"
    else:
        return "Low-Volume / Scalping Phase"

if __name__ == '__main__':
    print("Fetching latest 5-day market data for XRP/USDT...")
    market_data = fetch_recent_data(hours_back=120, timeframe='5m')
    
    print("\nAnalyzing volume cycle to determine current market state...")
    current_signal = get_dynamic_signal(market_data)
    
    print("\n" + "="*40)
    print(f"DYNAMIC SIGNAL: {current_signal}")
    print("="*40)