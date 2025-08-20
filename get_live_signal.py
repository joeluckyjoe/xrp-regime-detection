import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pygam import LinearGAM, s
from datetime import datetime, timedelta, timezone
import ccxt
import warnings
import pandas_ta as ta
from arch import arch_model

warnings.filterwarnings("ignore")

# --- Your optimized parameters from the final run ---
OPTIMIZED_PARAMS = {
    "sl_multiplier": 3.0,
    "tp_multiplier": 5.0,
    "short_window": 10,
    "long_window": 80,
    "rsi_oversold": 20,
    "rsi_overbought": 80
}

def fetch_live_data(hours_back=21, timeframe='5m'):
    """Fetches the most recent data needed for the model."""
    exchange = ccxt.binance()
    since_datetime = datetime.now(timezone.utc) - timedelta(hours=hours_back)
    since_timestamp = int(since_datetime.timestamp() * 1000)
    ohlcv = exchange.fetch_ohlcv('XRP/USDT', timeframe, since=since_timestamp, limit=300)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def generate_live_signal(data, params):
    """
    Generates a single, actionable signal for the current market state.
    """
    # 1. Determine current regime using GAM on volume
    log_volume = np.log(data['volume']).dropna()
    scaler_gam = StandardScaler()
    scaled_log_volume = scaler_gam.fit_transform(log_volume.values.reshape(-1, 1))
    time_idx = np.arange(len(scaled_log_volume))
    time_of_day = time_idx % 288
    gam = LinearGAM(s(0, n_splines=15, basis='cp')).fit(time_of_day, scaled_log_volume)
    current_prediction = gam.predict(time_of_day[-1:])
    cycle_mean = gam.predict(time_of_day).mean()
    regime = "high_volatility" if current_prediction[0] > cycle_mean else "low_volatility"

    # 2. Calculate necessary indicators
    data.ta.rsi(length=14, append=True)
    data.ta.bbands(length=params['short_window'], append=True, col_names=(f'BB_LOWER', f'BB_MIDDLE', f'BB_UPPER', 'BB_BW', 'BB_P'))
    data['ma_200'] = data['close'].rolling(window=200).mean()
    data.ta.ema(length=params['short_window'], append=True, col_names='EMA_short')
    data.ta.ema(length=params['long_window'], append=True, col_names='EMA_long')
    data.ta.atr(length=14, append=True, col_names='ATR_14')
    data.dropna(inplace=True)
    atr_threshold = data['ATR_14'].rolling(window=288).mean()
    
    # 3. Check for a signal at the most recent candle
    latest = data.iloc[-1]
    prev = data.iloc[-2]
    
    signal = "HOLD"
    if regime == 'low_volatility':
        if latest['close'] < latest['BB_LOWER'] and latest['close'] > latest['ma_200'] and latest['RSI_14'] < params['rsi_oversold']:
            signal = "BUY (Mean-Reversion)"
        elif latest['close'] > latest['BB_UPPER'] and latest['close'] < latest['ma_200'] and latest['RSI_14'] > params['rsi_overbought']:
            signal = "SELL (Mean-Reversion)"
            
    elif regime == 'high_volatility':
        if (prev['EMA_short'] < prev['EMA_long'] and latest['EMA_short'] > latest['EMA_long'] and
            latest['close'] > latest['ma_200'] and latest['ATR_14'] > atr_threshold.iloc[-1]):
            signal = "BUY (Momentum)"
        elif (prev['EMA_short'] > prev['EMA_long'] and latest['EMA_short'] < latest['EMA_long'] and
              latest['close'] < latest['ma_200'] and latest['ATR_14'] > atr_threshold.iloc[-1]):
            signal = "SELL (Momentum)"

    # 4. Calculate dynamic risk for the signal
    if signal != "HOLD":
        garch_returns = np.log(data['close'] / data['close'].shift(1)).dropna() * 100
        garch_model = arch_model(garch_returns, vol='Garch', p=1, q=1).fit(disp='off')
        forecast = garch_model.forecast(horizon=1)
        estimated_sigma = np.sqrt(forecast.variance.iloc[-1, 0]) / 100
        
        sl = estimated_sigma * params['sl_multiplier']
        tp = estimated_sigma * params['tp_multiplier']
        return signal, f"{sl:.2%}", f"{tp:.2%}"
        
    return signal, "N/A", "N/A"

if __name__ == '__main__':
    print("Fetching live market data...")
    live_data = fetch_live_data()
    
    print("Generating live trading signal with optimized parameters...")
    final_signal, stop_loss, take_profit = generate_live_signal(live_data, OPTIMIZED_PARAMS)
    
    print("\n" + "="*50)
    print("         REAL-TIME TRADING SIGNAL")
    print("="*50)
    print(f"Time:           {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Signal:         {final_signal}")
    print(f"Stop-Loss:      {stop_loss}")
    print(f"Take-Profit:    {take_profit}")
    print("="*50)