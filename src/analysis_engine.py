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

def fetch_data(start_str, end_str, timeframe='5m'):
    """Fetches a specific slice of historical OHLCV data."""
    exchange = ccxt.binance()
    start_dt = datetime.strptime(start_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    since_timestamp = int(start_dt.timestamp() * 1000)
    end_timestamp = int(end_dt.timestamp() * 1000)
    
    all_ohlcv = []
    while since_timestamp < end_timestamp:
        ohlcv = exchange.fetch_ohlcv('XRP/USDT', timeframe, since=since_timestamp, limit=1000)
        if len(ohlcv) == 0: break
        all_ohlcv.extend(ohlcv)
        since_timestamp = ohlcv[-1][0] + 1
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df.tz_localize('UTC')
    return df[(df.index >= start_dt) & (df.index < end_dt)]

def get_live_signal_and_risk(data, params):
    """The complete analysis engine. Takes data and parameters, returns a signal dictionary."""
    log_volume = np.log(data['volume']).dropna()
    scaler_gam = StandardScaler()
    scaled_log_volume = scaler_gam.fit_transform(log_volume.values.reshape(-1, 1))
    time_idx = np.arange(len(scaled_log_volume))
    time_of_day = time_idx % 288
    gam = LinearGAM(s(0, n_splines=15, basis='cp')).fit(time_of_day, scaled_log_volume)
    current_prediction = gam.predict(time_of_day[-1:])
    cycle_mean = gam.predict(time_of_day).mean()
    regime = "high_volatility" if current_prediction[0] > cycle_mean else "low_volatility"

    data.ta.rsi(length=14, append=True)
    data.ta.bbands(length=params['short_window'], append=True, col_names=(f'BB_LOWER', f'BB_MIDDLE', f'BB_UPPER', 'BB_BW', 'BB_P'))
    data['ma_200'] = data['close'].rolling(window=200).mean()
    data.ta.ema(length=params['short_window'], append=True, col_names='EMA_short')
    data.ta.ema(length=params['long_window'], append=True, col_names='EMA_long')
    data.ta.atr(length=14, append=True, col_names='ATR_14')
    data.dropna(inplace=True)
    atr_threshold = data['ATR_14'].rolling(window=288).mean()
    
    latest = data.iloc[-1]
    prev = data.iloc[-2]
    
    signal = "HOLD"
    signal_price = latest['close']
    reason = "No conditions met."

    if regime == 'low_volatility':
        if latest['close'] < latest['BB_LOWER'] and latest['close'] > latest['ma_200'] and latest['RSI_14'] < params['rsi_oversold']:
            signal = "BUY (Mean-Reversion)"
            reason = f"Price ({latest['close']:.4f}) crossed below Lower Bollinger Band ({latest['BB_LOWER']:.4f}) and RSI ({latest['RSI_14']:.1f}) is oversold."
        elif latest['close'] > latest['BB_UPPER'] and latest['close'] < latest['ma_200'] and latest['RSI_14'] > params['rsi_overbought']:
            signal = "SELL (Mean-Reversion)"
            reason = f"Price ({latest['close']:.4f}) crossed above Upper Bollinger Band ({latest['BB_UPPER']:.4f}) and RSI ({latest['RSI_14']:.1f}) is overbought."
            
    elif regime == 'high_volatility':
        if (prev['EMA_short'] < prev['EMA_long'] and latest['EMA_short'] > latest['EMA_long'] and latest['close'] > latest['ma_200'] and latest['ATR_14'] > atr_threshold.iloc[-1]):
            signal = "BUY (Momentum)"
            reason = f"Short EMA ({latest['EMA_short']:.4f}) crossed above Long EMA ({latest['EMA_long']:.4f}) in a high volatility environment."
        elif (prev['EMA_short'] > prev['EMA_long'] and latest['EMA_short'] < latest['EMA_long'] and latest['close'] < latest['ma_200'] and latest['ATR_14'] > atr_threshold.iloc[-1]):
            signal = "SELL (Momentum)"
            reason = f"Short EMA ({latest['EMA_short']:.4f}) crossed below Long EMA ({latest['EMA_long']:.4f}) in a high volatility environment."
            
    result = {"regime": regime, "signal": signal, "reason": reason, "latest_price": latest['close']}
            
    if signal != "HOLD":
        garch_returns = np.log(data['close'] / data['close'].shift(1)).dropna() * 100
        garch_model = arch_model(garch_returns.iloc[-250:], vol='Garch', p=1, q=1).fit(disp='off')
        forecast = garch_model.forecast(horizon=1)
        estimated_sigma = np.sqrt(forecast.variance.iloc[-1, 0]) / 100
        result["stop_loss_pct"] = estimated_sigma * params['sl_multiplier']
        result["take_profit_pct"] = estimated_sigma * params['tp_multiplier']
        result["signal_price"] = signal_price
        
    return result