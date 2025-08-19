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

def get_dynamic_risk_signal(data, short_window=20):
    """
    Analyzes the volume cycle and returns a complete signal with dynamic risk sizing.
    """
    # 1. Prepare the data
    log_volume = np.log(data['volume']).dropna()
    scaler = StandardScaler()
    scaled_log_volume = scaler.fit_transform(log_volume.values.reshape(-1, 1))
    time_idx = np.arange(len(scaled_log_volume))
    time_of_day = time_idx % 288

    # 2. Fit the GAM
    gam = LinearGAM(s(0, n_splines=15, basis='cp')).fit(time_of_day, scaled_log_volume)
    
    # 3. Get predictions and determine the regime
    predictions = gam.predict(time_of_day)
    cycle_mean = predictions.mean()
    current_prediction = predictions[-1]
    
    regime = "High-Volume / Momentum Phase" if current_prediction > cycle_mean else "Low-Volume / Scalping Phase"

    # 4. Calculate a confidence score (0 to 1)
    # This measures how far the current prediction is from the average
    confidence = abs(current_prediction - cycle_mean) / (predictions.max() - predictions.min())
    
    # 5. Determine position size based on confidence
    max_risk_per_trade = 2.0  # e.g., risk 2% of capital
    position_size_pct = max_risk_per_trade * confidence
    
    # 6. Check for tactical signals at the most recent data point
    latest_data = data.iloc[-1]
    bb_middle = latest_data['close-20d-sma'] # Assuming you pre-calculate this or get it from an API
    bb_std = latest_data['close-20d-std']
    bb_upper = bb_middle + (bb_std * 2)
    bb_lower = bb_middle - (bb_std * 2)

    tactical_signal = "None"
    if regime == "Low-Volume / Scalping Phase":
        if latest_data['close'] < bb_lower:
            tactical_signal = "BUY"
        elif latest_data['close'] > bb_upper:
            tactical_signal = "SELL"
            
    return regime, tactical_signal, confidence, position_size_pct

if __name__ == '__main__':
    print("Fetching latest market data...")
    market_data = fetch_recent_data(hours_back=120, timeframe='5m')
    
    # Pre-calculate indicators needed for the latest signal
    market_data['close-20d-sma'] = market_data['close'].rolling(window=20).mean()
    market_data['close-20d-std'] = market_data['close'].rolling(window=20).std()
    market_data.dropna(inplace=True)
    
    print("\nAnalyzing market state for dynamic signal...")
    current_regime, tactical_signal, confidence, risk_pct = get_dynamic_risk_signal(market_data)
    
    print("\n" + "="*50)
    print("         DYNAMIC TRADING & RISK SIGNAL")
    print("="*50)
    print(f"Current Time:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Detected Regime:     {current_regime}")
    print(f"Model Confidence:    {confidence:.2%}")
    print(f"Tactical Signal:     {tactical_signal}")
    print(f"Suggested Risk:      {risk_pct:.2f}% of trading capital")
    print("="*50)