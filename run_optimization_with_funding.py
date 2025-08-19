import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pygam import LinearGAM, s
from datetime import datetime, timedelta, timezone
import ccxt
import warnings
import pandas_ta as ta
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

# Suppress standard warnings for cleaner output
warnings.filterwarnings("ignore")

def fetch_recent_data(hours_back=360, timeframe='5m', exchange_name='binance'):
    """Fetches recent historical OHLCV data."""
    exchange = ccxt.binance()
    since_datetime = datetime.now(timezone.utc) - timedelta(hours=hours_back)
    since_timestamp = int(since_datetime.timestamp() * 1000)
    all_ohlcv = []
    while True:
        ohlcv = exchange.fetch_ohlcv('XRP/USDT', timeframe, since=since_timestamp, limit=1000)
        if len(ohlcv) == 0:
            break
        all_ohlcv.extend(ohlcv)
        since_timestamp = ohlcv[-1][0] + 1
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def fetch_funding_rates(since, timeframe='5m', exchange_name='binanceusdm'):
    """Fetches historical funding rate data."""
    exchange = getattr(ccxt, exchange_name)()
    since_timestamp = int(since.timestamp() * 1000)
    funding_data = exchange.fetch_funding_rate_history('XRP/USDT:USDT', since=since_timestamp, limit=1000)
    df = pd.DataFrame(funding_data)
    
    # CORRECTED LINE: Use the main 'timestamp' field instead of the nested one
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    df = df.set_index('timestamp')['fundingRate'].astype(float)
    return df.resample(timeframe).ffill()

def run_filtered_backtest(data, stop_loss_pct, take_profit_pct, short_window, long_window, rsi_oversold, rsi_overbought, funding_rate_threshold, transaction_cost=0.001):
    """
    The full backtesting function, now with a funding rate parameter.
    """
    # 1. Generate historical regimes
    log_volume = np.log(data['volume']).dropna()
    scaler = StandardScaler()
    scaled_log_volume = scaler.fit_transform(log_volume.values.reshape(-1, 1))
    time_idx = np.arange(len(scaled_log_volume))
    time_of_day = time_idx % 288
    gam = LinearGAM(s(0, n_splines=15, basis='cp')).fit(time_of_day, scaled_log_volume)
    predictions = gam.predict(time_of_day)
    cycle_mean = predictions.mean()
    
    df = data.loc[log_volume.index].copy()
    df['regime'] = np.where(predictions > cycle_mean, 'high_volatility', 'low_volatility')

    # 2. Calculate all necessary indicators
    df['ma_200'] = df['close'].rolling(window=200).mean()
    df.ta.rsi(length=14, append=True)
    df.ta.bbands(length=short_window, append=True, col_names=(f'BB_LOWER_{short_window}', f'BB_MIDDLE_{short_window}', f'BB_UPPER_{short_window}', f'BB_BANDWIDTH_{short_window}', f'BB_PERCENT_{short_window}'))
    df.ta.ema(length=short_window, append=True, col_names=f'EMA_{short_window}')
    df.ta.ema(length=long_window, append=True, col_names=f'EMA_{long_window}')
    df.ta.atr(length=14, append=True, col_names='ATR_14')
    df.dropna(inplace=True)
    
    atr_threshold = df['ATR_14'].rolling(window=288).mean()

    # 3. Simulate Trades
    position = 0
    pnl = 0
    entry_price = 0
    stop_loss_price = 0
    take_profit_price = 0
    
    for i in range(1, len(df)):
        current_row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        # ... (Exit logic is the same as before) ...
        
        if position == 0:
            current_funding_rate = current_row['fundingRate']
            if pd.notna(current_funding_rate):
                if current_row['regime'] == 'low_volatility':
                    if (current_row['close'] < current_row[f'BB_LOWER_{short_window}'] and 
                        current_row['close'] > current_row['ma_200'] and 
                        current_row['RSI_14'] < rsi_oversold and
                        current_funding_rate < funding_rate_threshold):
                        position = 1
                        entry_price = current_row['open']
                        stop_loss_price = entry_price * (1 - stop_loss_pct)
                        take_profit_price = entry_price * (1 + take_profit_pct)

    return pnl

# --- Bayesian Optimization Setup ---
param_space = [
    Real(0.001, 0.02, name='stop_loss_pct'),
    Real(0.002, 0.04, name='take_profit_pct'),
    Integer(10, 35, name='short_window'),
    Integer(40, 80, name='long_window'),
    Integer(20, 40, name='rsi_oversold'),
    Integer(60, 80, name='rsi_overbought'),
    Real(0.0001, 0.001, name='funding_rate_threshold'), # New parameter to optimize
]

if __name__ == '__main__':
    print("Fetching 6 months of market data for final optimization...")
    market_data = fetch_recent_data(hours_back=4320, timeframe='5m')
    print("Fetching funding rate data...")
    funding_rates = fetch_funding_rates(since=market_data.index[0], timeframe='5m')
    market_data = market_data.join(funding_rates)
    
    @use_named_args(param_space)
    def objective(**params):
        pnl = run_filtered_backtest(data=market_data.copy(), **params)
        return -pnl

    print("\nRunning Bayesian Optimization with Funding Rate filter...")
    print("This will run the backtest 50 times and may take several minutes.")
    
    result = gp_minimize(
        objective,
        param_space,
        n_calls=50,
        random_state=0,
        n_jobs=-1
    )

    best_pnl = -result.fun
    best_params = result.x

    print("\n" + "="*50)
    print("     BAYESIAN OPTIMIZATION (WITH FUNDING) COMPLETE")
    print("="*50)
    print(f"Best PNL Found:      ${best_pnl:.4f}")
    print("\nOptimal Parameters:")
    for param, value in zip(param_space, best_params):
        print(f"- {param.name}: {value}")
    print("="*50)