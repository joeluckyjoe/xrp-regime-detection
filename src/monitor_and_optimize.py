import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import warnings
import pymc as pm
import subprocess
import smtplib
from email.mime.text import MIMEText
import ccxt

warnings.filterwarnings("ignore")

EMAIL_CONFIG = {"smtp_server": "smtp.office365.com", "smtp_port": 587, "sender_email": "joeluckyjoe@hotmail.com", "sender_password": "hwzfapskhluofrnh", "recipient_email": "joeluckyjoe@hotmail.com"}
TRADE_LOG_FILE = 'data/live_trade_log.csv'
OPTIMIZATION_SCRIPT = 'run_definitive_optimization.py'
MONITORING_PERIOD_DAYS = 30

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

def calculate_forward_returns(trade_log):
    if len(trade_log) < 10: return pd.Series(dtype=np.float64)
    start_date, end_date = trade_log.index.min() - timedelta(days=1), trade_log.index.max() + timedelta(days=2)
    price_data = fetch_historical_data(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    returns = []
    for index, row in trade_log.iterrows():
        entry_time, entry_price = index, row['price']
        signal_type = 1 if 'BUY' in row['signal'] else -1
        exit_time = entry_time + timedelta(hours=24)
        future_prices = price_data[price_data.index > entry_time]
        if not future_prices.empty:
            exit_price_row = future_prices[future_prices.index >= exit_time]
            if not exit_price_row.empty:
                exit_price = exit_price_row['close'].iloc[0]
                returns.append(((exit_price - entry_price) / entry_price) * signal_type)
    return pd.Series(returns)

def check_for_structural_break(pnl_series):
    if len(pnl_series) < 20: return False
    with pm.Model() as model:
        mu = pm.Normal('mu', mu=0, sigma=0.01, shape=2)
        change_point = pm.DiscreteUniform('change_point', lower=1, upper=len(pnl_series)-1)
        idx = pm.math.switch(pm.math.ge(np.arange(len(pnl_series)), change_point), 1, 0)
        likelihood = pm.Normal('likelihood', mu=mu[idx], sigma=pnl_series.std(), observed=pnl_series)
        trace = pm.sample(2000, tune=1000, chains=2, cores=1, progressbar=False, step=pm.Metropolis())
    change_probs = np.bincount(trace.posterior['change_point'].values.flatten()) / len(trace.posterior['change_point'].values.flatten())
    most_likely_cp_idx = change_probs.argmax()
    is_recent = most_likely_cp_idx > (len(pnl_series) * 0.75)
    mean_perf_before, mean_perf_after = trace.posterior['mu'].values[:, :, 0].mean(), trace.posterior['mu'].values[:, :, 1].mean()
    performance_degraded = mean_perf_after < mean_perf_before
    return is_recent and performance_degraded

def run_optimization_in_background():
    print("Launching optimization in the background for the last 6 months...")
    end_date, start_date = datetime.now(timezone.utc), datetime.now(timezone.utc) - timedelta(days=180)
    start_date_str, end_date_str = start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
    command = f"nohup python {OPTIMIZATION_SCRIPT} {start_date_str} {end_date_str} > optimization.log 2>&1 &"
    subprocess.Popen(command, shell=True)
    print(f"Optimization process started. Check 'optimization.log' for progress.")

if __name__ == "__main__":
    print(f"--- Running Daily Strategy Monitor [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ---")
    try:
        trade_log = pd.read_csv(TRADE_LOG_FILE, header=None, names=['timestamp', 'signal', 'price'], index_col='timestamp', parse_dates=True)
        recent_log = trade_log[trade_log.index >= (datetime.now(timezone.utc) - timedelta(days=MONITORING_PERIOD_DAYS))]
        print(f"Found {len(recent_log)} trades in the last {MONITORING_PERIOD_DAYS} days.")
        pnl_series = calculate_forward_returns(recent_log)
        if not pnl_series.empty:
            needs_retraining = check_for_structural_break(pnl_series)
            if needs_retraining:
                print("RESULT: 🚨 Performance has degraded. Re-optimization is recommended.")
                send_email("Trading Strategy Alert: Performance Degraded", "An automatic re-optimization process has been launched.")
                run_optimization_in_background()
            else:
                print("RESULT: ✅ Strategy performance is stable. No action needed.")
        else:
            print("Not enough recent trade data to perform analysis.")
    except FileNotFoundError:
        print("Trade log not found. No action needed.")
    except Exception as e:
        print(f"An error occurred during monitoring: {e}")
        send_email("Trading Bot ERROR in Monitor", str(e))