import json
import time
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from src.analysis_engine import fetch_live_data, get_live_signal_and_risk

# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================
EMAIL_CONFIG = {
    "smtp_server": "smtp.office365.com",
    "smtp_port": 587,
    "sender_email": "joeluckyjoe@hotmail.com",
    "sender_password": "hwzfapskhluofrnh",
    "recipient_email": "joeluckyjoe@hotmail.com"
}
TRADING_CONFIG = {
    "total_capital_usd": 10000,
    "risk_per_trade_pct": 0.01
}
STATE_FILE = "data/last_signal.txt"

# ==============================================================================
# --- HELPER FUNCTIONS ---
# ==============================================================================
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

def get_last_signal():
    try:
        with open(STATE_FILE, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        return "HOLD"

def save_last_signal(signal):
    with open(STATE_FILE, 'w') as f:
        f.write(signal)

# ==============================================================================
# --- MAIN EXECUTION LOOP ---
# ==============================================================================
if __name__ == "__main__":
    print("--- Live Trading Bot Started ---")
    
    while True:
        try:
            # 1. Load parameters
            with open('parameters.json', 'r') as f:
                params = json.load(f)
            
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] --- Checking for new signal ---")
            
            # 2. Get signal from the engine
            live_data = fetch_live_data()
            signal_data = get_live_signal_and_risk(live_data, params)
            
            new_signal = signal_data["signal"]
            print(f"Current Regime: {signal_data['regime']}")
            print(f"Current Signal: {new_signal}")
            
            # 3. Compare with last known signal
            last_signal = get_last_signal()
            if new_signal != "HOLD" and new_signal != last_signal:
                print(f"!!! NEW SIGNAL DETECTED: {new_signal} !!!")
                
                # Calculate all order parameters
                capital_to_risk = TRADING_CONFIG["total_capital_usd"] * TRADING_CONFIG["risk_per_trade_pct"]
                position_size_usd = capital_to_risk / signal_data["stop_loss_pct"]
                position_size_xrp = position_size_usd / signal_data["signal_price"]
                
                if "BUY" in new_signal:
                    entry_price = signal_data["signal_price"]
                    stop_loss_price = entry_price * (1 - signal_data["stop_loss_pct"])
                    take_profit_price = entry_price * (1 + signal_data["take_profit_pct"])
                else: # SELL
                    entry_price = signal_data["signal_price"]
                    stop_loss_price = entry_price * (1 + signal_data["stop_loss_pct"])
                    take_profit_price = entry_price * (1 - signal_data["take_profit_pct"])

                # Compose and send the email
                email_subject = f"XRP/USDT Signal: {new_signal}"
                email_body = f"""
                --- NEW TRADING SIGNAL ---

                - ACTION: {new_signal}
                - STRATEGY: {signal_data['regime']}

                --- ORDER TICKET ---
                - Order Type: Limit
                - Entry Price: ${entry_price:,.4f}
                - Position Size (USD): ${position_size_usd:,.2f}
                - Position Size (XRP): {position_size_xrp:,.2f}
                - Stop-Loss Price: ${stop_loss_price:,.4f} ({signal_data['stop_loss_pct']:.2%})
                - Take-Profit Price: ${take_profit_price:,.4f} ({signal_data['take_profit_pct']:.2%})

                Note: This signal is time-sensitive. Consider cancelling the order if not filled within 15-30 minutes.
                """
                send_email(email_subject, email_body)
                save_last_signal(new_signal)
                
            elif new_signal == "HOLD":
                save_last_signal("HOLD")

        except Exception as e:
            print(f"An error occurred: {e}")
            send_email("Trading Bot ERROR", str(e))

        print("--- Waiting for 5 minutes ---")
        time.sleep(300) 