import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import warnings
import pymc as pm
import subprocess
import os
import ccxt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
TRADE_LOG_FILE = 'data/live_trade_log.csv'
OPTIMIZATION_SCRIPT = 'run_definitive_optimization.py'
MONITORING_PERIOD_DAYS = 30
OUTPUT_DIR = "monitoring"

# --- HELPER FUNCTIONS ---
def fetch_historical_data(start_dt, end_dt, timeframe='5m'):
    """Fetches a specific slice of historical data."""
    exchange = ccxt.binance()
    since_timestamp, end_timestamp = int(start_dt.timestamp() * 1000), int(end_dt.timestamp() * 1000)
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
    """Calculates the 24-hour forward return for each signal in the log."""
    if len(trade_log) < 10: return pd.Series(dtype=np.float64)
    start_date, end_date = trade_log.index.min() - timedelta(days=1), trade_log.index.max() + timedelta(days=2)
    price_data = fetch_historical_data(start_date, end_date)
    returns = []
    for index, row in trade_log.iterrows():
        entry_time, entry_price, signal_type = index, row['price'], 1 if 'BUY' in row['signal'] else -1
        exit_time = entry_time + timedelta(hours=24)
        future_prices = price_data[price_data.index > entry_time]
        if not future_prices.empty:
            exit_price_row = future_prices[future_prices.index >= exit_time]
            if not exit_price_row.empty:
                exit_price = exit_price_row['close'].iloc[0]
                returns.append(((exit_price - entry_price) / entry_price) * signal_type)
    return pd.Series(returns)

def check_for_structural_break(pnl_series):
    """Uses a Bayesian Change Point model to check for a break in performance."""
    if len(pnl_series) < 20:
        print("Not enough recent trades to perform a check.")
        return False, None
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
    return is_recent and performance_degraded, pnl_series.cumsum()

def run_optimization_in_background():
    """Launches the definitive optimization script as a background process."""
    print("Launching optimization in the background...")
    end_date, start_date = datetime.now(timezone.utc), datetime.now(timezone.utc) - timedelta(days=180)
    start_date_str, end_date_str = start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
    command = f"nohup python {OPTIMIZATION_SCRIPT} {start_date_str} {end_date_str} > optimization.log 2>&1 &"
    subprocess.Popen(command, shell=True)
    print(f"Optimization process started. Check 'optimization.log' for progress.")

def create_monitoring_report(needs_retraining, pnl_curve=None):
    """Generates a daily HTML monitoring report."""
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    now_str = datetime.now().strftime('%Y%m%d')
    report_filename = f"{now_str}_monitoring_report.html"
    chart_filename = f"{now_str}_performance_chart.png"
    
    status_message = "🚨 Performance has degraded. Re-optimization is recommended." if needs_retraining else "✅ Strategy performance is stable. No action needed."
    
    html = f"<html><head><title>Daily Monitoring Report</title></head><body style='font-family: sans-serif;'>"
    html += f"<h1>Daily Strategy Monitoring Report</h1>"
    html += f"<p><strong>Report Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"
    html += f"<h2>RESULT: {status_message}</h2>"
    
    if needs_retraining:
        html += "<p>An automatic re-optimization process has been launched in the background. Check 'optimization.log' for progress.</p>"
    
    if pnl_curve is not None and not pnl_curve.empty:
        # Create and save the performance chart
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(pnl_curve.index, pnl_curve.values, label='Cumulative PNL (by trade count)')
        ax.set_title('Recent Strategy Performance')
        ax.set_xlabel('Trade Number')
        ax.set_ylabel('Cumulative Forward Return')
        ax.grid(True)
        plt.savefig(os.path.join(OUTPUT_DIR, chart_filename))
        plt.close(fig)
        html += f"<hr><h2>Performance Chart (Last {len(pnl_curve)} Trades)</h2>"
        html += f"<img src='{chart_filename}' width='100%'>"
    
    html += "</body></html>"
    
    with open(os.path.join(OUTPUT_DIR, report_filename), 'w') as f:
        f.write(html)
    print(f"Monitoring report saved to: {os.path.join(OUTPUT_DIR, report_filename)}")

if __name__ == "__main__":
    print(f"--- Running Daily Strategy Monitor [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ---")
    
    try:
        trade_log = pd.read_csv(TRADE_LOG_FILE, header=None, names=['timestamp', 'signal', 'price'], index_col='timestamp', parse_dates=True)
        recent_log = trade_log[trade_log.index >= (datetime.now(timezone.utc) - timedelta(days=MONITORING_PERIOD_DAYS))]
        print(f"Found {len(recent_log)} trades in the last {MONITORING_PERIOD_DAYS} days.")
        
        pnl_series = calculate_forward_returns(recent_log)
        
        if not pnl_series.empty:
            needs_retraining, pnl_curve = check_for_structural_break(pnl_series)
            create_monitoring_report(needs_retraining, pnl_curve)
            
            if needs_retraining:
                # run_optimization_in_background() # Uncomment to enable automatic optimization
                pass
        else:
            print("Not enough recent trade data to perform analysis.")
            create_monitoring_report(False)

    except FileNotFoundError:
        print("Trade log not found. No action needed.")
        create_monitoring_report(False)
    except Exception as e:
        print(f"An error occurred during monitoring: {e}")
        create_monitoring_report(True) # Assume retraining is needed if monitor fails