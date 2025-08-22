import json
import time
from datetime import datetime, timedelta, timezone
import os
import pandas as pd
from src.analysis_engine import fetch_data, get_live_signal_and_risk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
TRADING_CONFIG = {"total_capital_usd": 10000, "risk_per_trade_pct": 0.01}
STATE_FILE = "data/last_signal.txt"
OUTPUT_DIR = "signals"
TRADE_LOG_FILE = "data/live_trade_log.csv"

def create_signal_report(data, signal_data, params):
    """Generates a chart and saves a detailed HTML report."""
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    chart_filename, report_filename = f"{now_str}_signal_chart.png", f"{now_str}_signal_report.html"
    
    # Create Chart
    fig, ax = plt.subplots(figsize=(15, 8))
    plot_data = data.iloc[-288:] # Plot last 24 hours
    ax.plot(plot_data.index, plot_data['close'], label='Close Price', color='k', alpha=0.7)
    signal_price = signal_data['signal_price']
    if "Mean-Reversion" in signal_data['signal']:
        ax.plot(plot_data.index, plot_data[f'BB_LOWER_{params["short_window"]}'], 'b--', label='Bollinger Bands')
        ax.plot(plot_data.index, plot_data[f'BB_UPPER_{params["short_window"]}'], 'b--')
        ax.plot(plot_data.index[-1], signal_price, '^' if 'BUY' in signal_data['signal'] else 'v', markersize=15, color='g' if 'BUY' in signal_data['signal'] else 'r', label='Signal')
    elif "Momentum" in signal_data['signal']:
        ax.plot(plot_data.index, plot_data[f'EMA_{params["short_window"]}'], 'C0', label='Short EMA')
        ax.plot(plot_data.index, plot_data[f'EMA_{params["long_window"]}'], 'C1', label='Long EMA')
    ax.set_title(f"Signal: {signal_data['signal']} at {signal_price:.4f}")
    ax.legend(), ax.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, chart_filename)), plt.close(fig)

    # Create HTML Report
    capital_to_risk = TRADING_CONFIG["total_capital_usd"] * TRADING_CONFIG["risk_per_trade_pct"]
    position_size_usd = capital_to_risk / signal_data["stop_loss_pct"]
    position_size_xrp = position_size_usd / signal_price
    if "BUY" in signal_data['signal']:
        stop_loss_price, take_profit_price = signal_price * (1 - signal_data["stop_loss_pct"]), signal_price * (1 + signal_data["take_profit_pct"])
    else:
        stop_loss_price, take_profit_price = signal_price * (1 + signal_data["stop_loss_pct"]), signal_price * (1 - signal_data["take_profit_pct"])
        
    html = f"""
    <html><head><title>Trading Signal</title></head><body style="font-family: sans-serif;">
        <h1>New Trading Signal Detected</h1>
        <p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Signal:</strong> {signal_data['signal']}</p>
        <p><strong>Reason:</strong> {signal_data['reason']}</p>
        <p><strong>Regime:</strong> {signal_data['regime']}</p>
        <hr>
        <h2>Order Ticket</h2>
        <table border="1" cellpadding="5">
            <tr><td>Order Type</td><td>Limit</td></tr>
            <tr><td>Entry Price</td><td>${signal_price:,.4f}</td></tr>
            <tr><td>Position Size (USD)</td><td>${position_size_usd:,.2f}</td></tr>
            <tr><td>Position Size (XRP)</td><td>{position_size_xrp:,.2f}</td></tr>
            <tr><td>Stop-Loss Price</td><td>${stop_loss_price:,.4f} ({signal_data['stop_loss_pct']:.2%})</td></tr>
            <tr><td>Take-Profit Price</td><td>${take_profit_price:,.4f} ({signal_data['take_profit_pct']:.2%})</td></tr>
        </table>
        <p><i>Note: This signal is time-sensitive. Consider cancelling the order if not filled within 15-30 minutes.</i></p>
        <hr>
        <h2>Signal Chart</h2>
        <img src='{chart_filename}' width='100%'>
    </body></html>
    """
    with open(os.path.join(OUTPUT_DIR, report_filename), 'w') as f:
        f.write(html)
    print(f"Detailed report saved to: {os.path.join(OUTPUT_DIR, report_filename)}")

def get_last_signal():
    try:
        with open(STATE_FILE, 'r') as f: return f.read().strip()
    except FileNotFoundError: return "HOLD"

def save_last_signal(signal):
    with open(STATE_FILE, 'w') as f: f.write(signal)

if __name__ == "__main__":
    print("--- Live Trading Bot Started (File Logging Enabled) ---")
    while True:
        try:
            with open('parameters.json', 'r') as f: params = json.load(f)
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] --- Checking for new signal ---")
            
            end_date, start_date = datetime.now(timezone.utc), datetime.now(timezone.utc) - timedelta(days=2)
            live_data = fetch_data(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            
            signal_data = get_live_signal_and_risk(live_data, params)
            new_signal = signal_data["signal"]
            print(f"Current Signal: {new_signal}")
            
            last_signal = get_last_signal()
            if new_signal != "HOLD" and new_signal != last_signal:
                print(f"!!! NEW SIGNAL DETECTED: {new_signal} !!!")
                create_signal_report(live_data, signal_data, params)
                
                # --- THIS LINE IS NOW RESTORED ---
                # Log the trade for the monitor to analyze later
                with open(TRADE_LOG_FILE, 'a') as log_file:
                    log_file.write(f"{datetime.now(timezone.utc).isoformat()},{new_signal},{signal_data['signal_price']}\n")
                
                save_last_signal(new_signal)
                
            elif new_signal == "HOLD":
                save_last_signal("HOLD")
        except Exception as e:
            print(f"An error occurred: {e}")
        print("--- Waiting for 5 minutes ---")
        time.sleep(300)