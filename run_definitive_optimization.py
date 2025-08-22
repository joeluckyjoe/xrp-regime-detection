import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pygam import LinearGAM, s
from datetime import datetime, timedelta, timezone
import ccxt
import warnings
import pandas_ta as ta
from arch import arch_model
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import sys
import json
import smtplib
from email.mime.text import MIMEText

warnings.filterwarnings("ignore")

# --- Email Configuration for sending the final alert ---
EMAIL_CONFIG = {
    "smtp_server": "smtp.office365.com",
    "smtp_port": 587,
    "sender_email": "joeluckyjoe@hotmail.com",
    "sender_password": "hwzfapskhluofrnh",
    "recipient_email": "joeluckyjoe@hotmail.com"
}

def send_email(subject, body):
    try:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = EMAIL_CONFIG["sender_email"]
        msg['To'] = EMAIL_CONFIG["recipient_email"]
        with smtplib.SMTP(EMAIL_CONFIG["smtp_server"], EMAIL_CONFIG["smtp_port"]) as server:
            server.starttls()
            server.login(EMAIL_CONFIG["sender_email"], EMAIL_CONFIG["sender_password"])
            server.send_message(msg)
        print(f"Email alert sent successfully: {subject}")
    except Exception as e:
        print(f"Error sending email: {e}")

def fetch_historical_data(start_date_str, end_date_str, timeframe='5m', exchange_name='binance'):
    exchange = ccxt.binance()
    start_dt = datetime.strptime(start_date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
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

def fetch_funding_rates(since, timeframe='5m', exchange_name='binanceusdm'):
    exchange = getattr(ccxt, exchange_name)()
    since_timestamp = int(since.timestamp() * 1000)
    all_funding = []
    while True:
        funding_data = exchange.fetch_funding_rate_history('XRP/USDT:USDT', since=since_timestamp, limit=1000)
        if len(funding_data) == 0: break
        all_funding.extend(funding_data)
        since_timestamp = funding_data[-1]['timestamp'] + 1
    df = pd.DataFrame(all_funding)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp')['fundingRate'].astype(float)
    df = df.tz_localize('UTC')
    return df.resample(timeframe).ffill()

def run_full_system_backtest(data, sl_multiplier, tp_multiplier, short_window, long_window, rsi_oversold, rsi_overbought, funding_rate_threshold, transaction_cost=0.00040):
    log_volume = np.log(data['volume']).replace([np.inf, -np.inf], np.nan).dropna()
    scaler_gam = StandardScaler()
    scaled_log_volume = scaler_gam.fit_transform(log_volume.values.reshape(-1, 1))
    time_idx = np.arange(len(scaled_log_volume))
    time_of_day = time_idx % 288
    gam = LinearGAM(s(0, n_splines=15, basis='cp')).fit(time_of_day, scaled_log_volume)
    predictions = gam.predict(time_of_day)
    cycle_mean = predictions.mean()
    df = data.loc[log_volume.index].copy()
    df['regime'] = np.where(predictions > cycle_mean, 'high_volatility', 'low_volatility')
    df['ma_200'] = df['close'].rolling(window=200).mean()
    df.ta.rsi(length=14, append=True)
    df.ta.bbands(length=int(short_window), append=True, col_names=(f'BB_LOWER_{int(short_window)}', f'BB_MIDDLE_{int(short_window)}', f'BB_UPPER_{int(short_window)}', 'BB_BANDWIDTH', 'BB_PERCENT'))
    df.ta.ema(length=int(short_window), append=True, col_names=f'EMA_{int(short_window)}')
    df.ta.ema(length=int(long_window), append=True, col_names=f'EMA_{int(long_window)}')
    df.ta.atr(length=14, append=True, col_names='ATR_14')
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df.dropna(inplace=True)
    atr_threshold = df['ATR_14'].rolling(window=288).mean()
    position, pnl, entry_price, stop_loss_price, take_profit_price = 0, 0, 0, 0, 0
    for i in range(250, len(df)):
        current_row, prev_row = df.iloc[i], df.iloc[i-1]
        if position != 0:
            exit_price = 0
            if position == 1 and (current_row['low'] <= stop_loss_price or current_row['high'] >= take_profit_price):
                exit_price = stop_loss_price if current_row['low'] <= stop_loss_price else take_profit_price
            elif position == -1 and (current_row['high'] >= stop_loss_price or current_row['low'] <= take_profit_price):
                exit_price = stop_loss_price if current_row['high'] >= stop_loss_price else take_profit_price
            if exit_price != 0:
                pnl += (exit_price - entry_price) * position - (abs(exit_price) + abs(entry_price)) * transaction_cost
                position = 0
        if position == 0:
            current_funding_rate, current_atr_threshold = current_row.get('fundingRate'), atr_threshold.get(current_row.name)
            if pd.notna(current_funding_rate) and pd.notna(current_atr_threshold):
                signal_generated = False
                if current_row['regime'] == 'low_volatility':
                    if (current_row['close'] < current_row[f'BB_LOWER_{int(short_window)}'] and current_row['close'] > current_row['ma_200'] and current_row['RSI_14'] < rsi_oversold and current_funding_rate < funding_rate_threshold):
                        position, signal_generated = 1, True
                    elif (current_row['close'] > current_row[f'BB_UPPER_{int(short_window)}'] and current_row['close'] < current_row['ma_200'] and current_row['RSI_14'] > rsi_overbought and current_funding_rate > -funding_rate_threshold):
                        position, signal_generated = -1, True
                elif current_row['regime'] == 'high_volatility':
                    if (prev_row[f'EMA_{int(short_window)}'] < prev_row[f'EMA_{int(long_window)}'] and current_row[f'EMA_{int(short_window)}'] > current_row[f'EMA_{int(long_window)}'] and current_row['close'] > current_row['ma_200'] and current_row['ATR_14'] > current_atr_threshold and current_funding_rate < funding_rate_threshold):
                        position, signal_generated = 1, True
                    elif (prev_row[f'EMA_{int(short_window)}'] > prev_row[f'EMA_{int(long_window)}'] and current_row[f'EMA_{int(short_window)}'] < current_row[f'EMA_{long_window}'] and current_row['close'] < current_row['ma_200'] and current_row['ATR_14'] > current_atr_threshold and current_funding_rate > -funding_rate_threshold):
                        position, signal_generated = -1, True
                if signal_generated:
                    entry_price = current_row['open']
                    garch_returns = df['log_returns'].iloc[i-250:i] * 100
                    garch_model = arch_model(garch_returns, vol='Garch', p=1, q=1).fit(disp='off')
                    forecast = garch_model.forecast(horizon=1)
                    estimated_sigma = np.sqrt(forecast.variance.iloc[-1, 0]) / 100
                    stop_loss_pct, take_profit_pct = estimated_sigma * sl_multiplier, estimated_sigma * tp_multiplier
                    if position == 1:
                        stop_loss_price, take_profit_price = entry_price * (1 - stop_loss_pct), entry_price * (1 + take_profit_pct)
                    else:
                        stop_loss_price, take_profit_price = entry_price * (1 + stop_loss_pct), entry_price * (1 - take_profit_pct)
    return pnl

param_space = [Real(1.0, 3.0, name='sl_multiplier'), Real(1.5, 5.0, name='tp_multiplier'), Integer(10, 35, name='short_window'), Integer(40, 80, name='long_window'), Integer(20, 40, name='rsi_oversold'), Integer(60, 80, name='rsi_overbought'), Real(0.0001, 0.001, name='funding_rate_threshold')]

if __name__ == '__main__':
    if len(sys.argv) == 3:
        start_date, end_date = sys.argv[1], sys.argv[2]
    else:
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=180)
        start_date, end_date = start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
    
    print(f"Fetching market data ({start_date} to {end_date})...")
    market_data = fetch_historical_data(start_date_str=start_date, end_date_str=end_date)
    print("Fetching funding rate data...")
    funding_rates = fetch_funding_rates(since=market_data.index[0], timeframe='5m')
    market_data = pd.merge_asof(left=market_data.sort_index(), right=funding_rates.sort_index(), left_index=True, right_index=True, direction='forward')
    
    @use_named_args(param_space)
    def objective(**params):
        pnl = run_full_system_backtest(data=market_data.copy(), **params)
        return -pnl

    print("\nRunning definitive Bayesian Optimization...")
    print("This will be very slow and will take several hours or more.")
    result = gp_minimize(objective, param_space, n_calls=50, random_state=0, n_jobs=-1)
    best_pnl = -result.fun
    best_params_list = result.x
    best_params_dict = {param.name: value for param, value in zip(param_space, best_params_list)}

    # --- THIS IS THE FIX ---
    # Convert any NumPy types to standard Python types before saving
    for key, value in best_params_dict.items():
        if isinstance(value, np.integer):
            best_params_dict[key] = int(value)
        elif isinstance(value, np.floating):
            best_params_dict[key] = float(value)

    print("\nOptimization complete. Updating parameters.json...")
    with open('parameters.json', 'w') as f:
        json.dump(best_params_dict, f, indent=4)
    
    email_subject = "Trading Strategy Optimization Complete"
    email_body = f"""
    The daily optimization process has completed.
    
    Best PNL Found: ${best_pnl:.4f}
    
    The following new parameters have been automatically deployed to parameters.json:
    {json.dumps(best_params_dict, indent=4)}
    
    The live bot will now use these new parameters.
    """
    send_email(email_subject, email_body)
    
    print("\n" + "="*50)
    print("         DEFINITIVE OPTIMIZATION COMPLETE")
    print("="*50)
    print(f"Best PNL Found:      ${best_pnl:.4f}")
    print("\nOptimal Parameters:")
    for name, value in best_params_dict.items():
        print(f"- {name}: {value}")
    print("="*50)