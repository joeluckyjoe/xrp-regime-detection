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

def fetch_historical_data(start_date_str, end_date_str, timeframe='5m', exchange_name='binance'):
    """Fetches a specific slice of historical data."""
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
    return df[start_date_str:end_date_str]

def run_full_system_backtest(data, sl_multiplier, tp_multiplier, short_window, long_window, rsi_oversold, rsi_overbought, funding_rate_threshold=0.001, transaction_cost=0.001):
    print(f"\n--- Starting Backtest ---")
    print(f"Initial data points: {len(data)}")

    # 1. Generate historical regimes
    log_volume = np.log(data['volume']).replace([np.inf, -np.inf], np.nan).dropna()
    df = data.loc[log_volume.index].copy()
    print(f"Data points after handling log(volume): {len(df)}")
    
    scaler_gam = StandardScaler()
    scaled_log_volume = scaler_gam.fit_transform(log_volume.values.reshape(-1, 1))
    time_idx = np.arange(len(scaled_log_volume))
    time_of_day = time_idx % 288
    gam = LinearGAM(s(0, n_splines=15, basis='cp')).fit(time_of_day, scaled_log_volume)
    predictions = gam.predict(time_of_day)
    cycle_mean = predictions.mean()
    df['regime'] = np.where(predictions > cycle_mean, 'high_volatility', 'low_volatility')

    # 2. Calculate indicators
    df['ma_200'] = df['close'].rolling(window=200).mean()
    df.ta.rsi(length=14, append=True)
    df.ta.bbands(length=short_window, append=True, col_names=(f'BB_LOWER_{short_window}', f'BB_MIDDLE_{short_window}', f'BB_UPPER_{short_window}', 'BB_BANDWIDTH', 'BB_PERCENT'))
    df.ta.ema(length=short_window, append=True, col_names=f'EMA_{short_window}')
    df.ta.ema(length=long_window, append=True, col_names=f'EMA_{long_window}')
    df.ta.atr(length=14, append=True, col_names='ATR_14')
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    print(f"Data points before dropping NaNs: {len(df)}")
    df.dropna(inplace=True)
    print(f"Data points after dropping NaNs: {len(df)}")
    
    if len(df) < 250:
        print("!!! ERROR: Not enough data remaining to run GARCH model. Exiting backtest. !!!")
        return 0.0, pd.DataFrame()
    
    atr_threshold = df['ATR_14'].rolling(window=288).mean()

    # 3. Simulate Trades
    position = 0
    pnl = 0
    trade_log = []
    
    # This loop is simplified for debugging purposes
    # A full loop would go here in the final version
    
    return pnl, pd.DataFrame(trade_log)

if __name__ == '__main__':
    start_date = '2025-02-20'
    end_date = '2025-08-20'
    print(f"Fetching 6 months of market data ({start_date} to {end_date})...")
    market_data = fetch_historical_data(start_date_str=start_date, end_date_str=end_date)
    
    print("\nRunning debug backtest...")
    # Using placeholder parameters for the debug run
    total_pnl, trades = run_full_system_backtest(
        market_data,
        sl_multiplier=1.5,
        tp_multiplier=3.0,
        short_window=20,
        long_window=50,
        rsi_oversold=30,
        rsi_overbought=70
    )
    
    print("\n" + "="*50)
    print("           DEBUG RUN COMPLETE")
    print("="*50)
    print(f"Total PNL: {total_pnl}")
    print(f"Total Trades: {len(trades)}")
    print("="*50)