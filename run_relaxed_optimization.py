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

# Suppress standard warnings for cleaner output
warnings.filterwarnings("ignore")

def fetch_recent_data(hours_back=360, timeframe='5m', exchange_name='binance'):
    """Fetches recent historical OHLCV data (15 days for a robust optimization)."""
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

def run_full_system_backtest(data, sl_multiplier, tp_multiplier, short_window, long_window, rsi_oversold, rsi_overbought, transaction_cost=0.001):
    """
    The full backtesting function, parameterized for optimization.
    """
    # 1. Generate historical regimes with GAM
    log_volume = np.log(data['volume']).dropna()
    scaler_gam = StandardScaler()
    scaled_log_volume = scaler_gam.fit_transform(log_volume.values.reshape(-1, 1))
    time_idx = np.arange(len(scaled_log_volume))
    time_of_day = time_idx % 288
    gam = LinearGAM(s(0, n_splines=15, basis='cp')).fit(time_of_day, scaled_log_volume)
    predictions = gam.predict(time_of_day)
    cycle_mean = predictions.mean()
    
    df = data.loc[log_volume.index].copy()
    df['regime'] = np.where(predictions > cycle_mean, 'high_volatility', 'low_volatility')

    # 2. Calculate all necessary technical indicators
    df['ma_200'] = df['close'].rolling(window=200).mean()
    df.ta.rsi(length=14, append=True)
    df.ta.bbands(length=short_window, append=True, col_names=(f'BB_LOWER_{short_window}', f'BB_MIDDLE_{short_window}', f'BB_UPPER_{short_window}', 'BB_BANDWIDTH', 'BB_PERCENT'))
    df.ta.ema(length=short_window, append=True, col_names=f'EMA_{short_window}')
    df.ta.ema(length=long_window, append=True, col_names=f'EMA_{long_window}')
    df.ta.atr(length=14, append=True, col_names='ATR_14')
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df.dropna(inplace=True)
    
    atr_threshold = df['ATR_14'].rolling(window=288).mean()

    # 3. Simulate Trades
    position = 0
    pnl = 0
    entry_price = 0
    stop_loss_price = 0
    take_profit_price = 0
    
    for i in range(250, len(df)):
        current_row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        # Exit logic
        if position == 1:
            if current_row['low'] <= stop_loss_price or current_row['high'] >= take_profit_price:
                exit_price = stop_loss_price if current_row['low'] <= stop_loss_price else take_profit_price
                pnl += (exit_price - entry_price) - (exit_price + entry_price) * transaction_cost
                position = 0
        elif position == -1:
            if current_row['high'] >= stop_loss_price or current_row['low'] <= take_profit_price:
                exit_price = stop_loss_price if current_row['high'] >= stop_loss_price else take_profit_price
                pnl += (entry_price - exit_price) - (entry_price + exit_price) * transaction_cost
                position = 0
        
        # Entry logic
        if position == 0:
            garch_returns = df['log_returns'].iloc[i-250:i] * 100
            garch_model = arch_model(garch_returns, vol='Garch', p=1, q=1).fit(disp='off')
            forecast = garch_model.forecast(horizon=1)
            estimated_sigma = np.sqrt(forecast.variance.iloc[-1, 0]) / 100
            
            stop_loss_pct = estimated_sigma * sl_multiplier
            take_profit_pct = estimated_sigma * tp_multiplier
            
            if current_row['regime'] == 'low_volatility':
                if (current_row['close'] < current_row[f'BB_LOWER_{short_window}'] and 
                    current_row['close'] > current_row['ma_200'] and 
                    current_row['RSI_14'] < rsi_oversold):
                    position = 1
                    entry_price = current_row['open']
                    stop_loss_price = entry_price * (1 - stop_loss_pct)
                    take_profit_price = entry_price * (1 + take_profit_pct)
                elif (current_row['close'] > current_row[f'BB_UPPER_{short_window}'] and
                      current_row['close'] < current_row['ma_200'] and 
                      current_row['RSI_14'] > rsi_overbought):
                    position = -1
                    entry_price = current_row['open']
                    stop_loss_price = entry_price * (1 + stop_loss_pct)
                    take_profit_price = entry_price * (1 - take_profit_pct)
            
            elif current_row['regime'] == 'high_volatility':
                if (prev_row[f'EMA_{short_window}'] < prev_row[f'EMA_{long_window}'] and 
                    current_row[f'EMA_{short_window}'] > current_row[f'EMA_{long_window}'] and
                    current_row['close'] > current_row['ma_200'] and
                    current_row['ATR_14'] > atr_threshold.iloc[i]):
                    position = 1
                    entry_price = current_row['open']
                    stop_loss_price = entry_price * (1 - stop_loss_pct)
                    take_profit_price = entry_price * (1 + take_profit_pct)
                elif (prev_row[f'EMA_{short_window}'] > prev_row[f'EMA_{long_window}'] and 
                      current_row[f'EMA_{short_window}'] < current_row[f'EMA_{long_window}'] and
                      current_row['close'] < current_row['ma_200'] and
                      current_row['ATR_14'] > atr_threshold.iloc[i]):
                    position = -1
                    entry_price = current_row['open']
                    stop_loss_price = entry_price * (1 + stop_loss_pct)
                    take_profit_price = entry_price * (1 - take_profit_pct)
    return pnl

# --- Bayesian Optimization Setup with RELAXED Parameters ---
param_space = [
    Real(0.5, 3.0, name='sl_multiplier'),      # Widen the search for SL
    Real(1.0, 5.0, name='tp_multiplier'),      # Widen the search for TP
    Integer(10, 35, name='short_window'),
    Integer(40, 80, name='long_window'),
    Integer(25, 45, name='rsi_oversold'),      # Relaxed: Allow buys at higher RSI
    Integer(55, 75, name='rsi_overbought'),    # Relaxed: Allow sells at lower RSI
]

if __name__ == '__main__':
    print("Fetching 15 days of market data for final optimization...")
    market_data = fetch_recent_data(hours_back=360, timeframe='5m')
    
    @use_named_args(param_space)
    def objective(**params):
        pnl = run_full_system_backtest(data=market_data.copy(), **params)
        return -pnl

    print("\nRunning final relaxed Bayesian Optimization...")
    print("This will run the backtest 50 times to find the best balance.")
    
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
    print("         RELAXED OPTIMIZATION COMPLETE")
    print("="*50)
    print(f"Best PNL Found:      ${best_pnl:.4f}")
    print("\nOptimal Parameters:")
    for param, value in zip(param_space, best_params):
        print(f"- {param.name}: {value}")
    print("="*50)