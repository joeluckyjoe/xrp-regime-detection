import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pygam import LinearGAM, s
from datetime import datetime, timedelta, timezone
import ccxt
import warnings
import pandas_ta as ta
from arch import arch_model
import time
import smtplib
from email.mime.text import MIMEText

# Suppress standard warnings for cleaner output
warnings.filterwarnings("ignore")

# ==============================================================================
# --- 1. CONFIGURATION ---
# ==============================================================================
# --- Paste your winning parameters from the optimization here ---
OPTIMIZED_PARAMS = {
    "sl_multiplier": 3.0,
    "tp_multiplier": 5.0,
    "short_window": 12,
    "long_window": 80,
    "rsi_oversold": 20,
    "rsi_overbought": 76,
    "funding_rate_threshold": 0.0001
}

# --- Email Configuration ---
SMTP_SERVER = "smtp.office365.com"
SMTP_PORT = 587
SENDER_EMAIL = "joeluckyjoe@hotmail.com"  # Your email
SENDER_PASSWORD = "hwzfapskhluofrnh" # Your App Password
RECIPIENT_EMAIL = "joeluckyjoe@hotmail.com  " # Where to send the alert

# ==============================================================================
# --- 2. HELPER FUNCTIONS ---
# ==============================================================================
def fetch_live_data(hours_back=48, timeframe='5m'): # Fetch last 2 days for indicators
    exchange = ccxt.binance()
    since_datetime = datetime.now(timezone.utc) - timedelta(hours=hours_back)
    since_timestamp = int(since_datetime.timestamp() * 1000)
    all_ohlcv = []
    while True:
        ohlcv = exchange.fetch_ohlcv('XRP/USDT', timeframe, since=since_timestamp, limit=1000)
        if len(ohlcv) == 0: break
        all_ohlcv.extend(ohlcv)
        since_timestamp = ohlcv[-1][0] + 1
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def send_email(subject, body):
    """Sends an email alert."""
    try:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECIPIENT_EMAIL
        
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
        print(f"Email alert sent successfully: {subject}")
    except Exception as e:
        print(f"Error sending email: {e}")

def get_live_signal(data, params):
    """Generates a single, actionable signal for the current market state."""
    # This is a simplified version of your backtest function, focused on the most recent data point
    # In a production system, you would refactor the backtest logic into a reusable class or module.
    
    # 1. Determine current regime
    log_volume = np.log(data['volume']).dropna()
    scaler_gam = StandardScaler()
    scaled_log_volume = scaler_gam.fit_transform(log_volume.values.reshape(-1, 1))
    time_idx = np.arange(len(scaled_log_volume))
    time_of_day = time_idx % 288
    gam = LinearGAM(s(0, n_splines=15, basis='cp')).fit(time_of_day, scaled_log_volume)
    current_prediction = gam.predict(time_of_day[-1:])
    cycle_mean = gam.predict(time_of_day).mean()
    regime = "high_volatility" if current_prediction[0] > cycle_mean else "low_volatility"
    
    # 2. Calculate indicators
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
        if (prev['EMA_short'] < prev['EMA_long'] and latest['EMA_short'] > latest['EMA_long'] and latest['close'] > latest['ma_200'] and latest['ATR_14'] > atr_threshold.iloc[-1]):
            signal = "BUY (Momentum)"
        elif (prev['EMA_short'] > prev['EMA_long'] and latest['EMA_short'] < latest['EMA_long'] and latest['close'] < latest['ma_200'] and latest['ATR_14'] > atr_threshold.iloc[-1]):
            signal = "SELL (Momentum)"
            
    # 4. Calculate dynamic risk for the signal
    if signal != "HOLD":
        garch_returns = np.log(data['close'] / data['close'].shift(1)).dropna() * 100
        garch_model = arch_model(garch_returns, vol='Garch', p=1, q=1).fit(disp='off')
        forecast = garch_model.forecast(horizon=1)
        estimated_sigma = np.sqrt(forecast.variance.iloc[-1, 0]) / 100
        sl = estimated_sigma * params['sl_multiplier']
        tp = estimated_sigma * params['tp_multiplier']
        return regime, signal, f"{sl:.2%}", f"{tp:.2%}"
        
    return regime, signal, "N/A", "N/A"

# ==============================================================================
# --- 3. MAIN EXECUTION LOOP ---
# ==============================================================================
if __name__ == "__main__":
    last_signal = "HOLD"
    
    while True:
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] --- Checking for new signal ---")
        
        # Fetch and analyze the latest data
        live_data = fetch_live_data()
        current_regime, new_signal, stop_loss, take_profit = get_live_signal(live_data, OPTIMIZED_PARAMS)
        
        print(f"Current Regime: {current_regime}")
        print(f"Current Signal: {new_signal}")
        
        # If a new, actionable signal is found, send an email
        if new_signal != "HOLD" and new_signal != last_signal:
            print(f"!!! NEW SIGNAL DETECTED: {new_signal} !!!")
            email_subject = f"XRP/USDT Trading Signal: {new_signal}"
            email_body = f"""
            New Trading Signal Detected:
            --------------------------------
            Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            Asset: XRP/USDT
            Signal: {new_signal}
            
            Market Conditions:
            --------------------------------
            Detected Regime: {current_regime}
            Latest Price: ${live_data['close'].iloc[-1]:.4f}
            
            Dynamic Risk Parameters:
            --------------------------------
            Stop-Loss: {stop_loss}
            Take-Profit: {take_profit}
            """
            send_email(email_subject, email_body)
            last_signal = new_signal
        elif new_signal == "HOLD":
            last_signal = "HOLD"

        # Wait for the next 5-minute candle
        print("--- Waiting for 5 minutes ---")
        time.sleep(300)