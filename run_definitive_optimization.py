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

def fetch_funding_rates(since, timeframe='5m', exchange_name='binanceusdm'):
    """Fetches historical funding rate data, looping to get all data."""
    exchange = getattr(ccxt, exchange_name)()
    since_timestamp = int(since.timestamp() * 1000)
    all_funding = []
    while True:
        funding_data = exchange.fetch_funding_rate_history('XRP/USDT:USDT', since=since_timestamp, limit=1000)
        if len(funding_data) == 0:
            break
        all_funding.extend(funding_data)
        # TYPO FIX: Changed funding__data to funding_data
        since_timestamp = funding_data[-1]['timestamp'] + 1
        
    df = pd.DataFrame(all_funding)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp')['fundingRate'].astype(float)
    return df.resample(timeframe).ffill()

def run_full_system_backtest(data, sl_multiplier, tp_multiplier, short_window, long_window, rsi_oversold, rsi_overbought, funding_rate_threshold, transaction_cost=0.001):
    """The full backtesting function with the complete simulation loop."""
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
    df.ta.bbands(length=short_window, append=True, col_names=(f'BB_LOWER_{short_window}', f'BB_MIDDLE_{short_window}', f'BB_UPPER_{short_window}', 'BB_BANDWIDTH', 'BB_PERCENT'))
    df.ta.ema(length=short_window, append=True, col_names=f'EMA_{short_window}')
    df.ta.ema(length=long_window, append=True, col_names=f'EMA_{long_window}')
    df.ta.atr(length=14, append=True, col_names='ATR_14')
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df.dropna(inplace=True)
    
    atr_threshold = df['ATR_14'].rolling(window=288).mean()

    position = 0
    pnl = 0
    entry_price = 0
    stop_loss_price = 0
    take_profit_price = 0
    
    for i in range(250, len(df)):
        current_row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
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
        
        if position == 0:
            current_funding_rate = current_row.get('fundingRate')
            if pd.notna(current_funding_rate):
                signal_generated = False
                
                if current_row['regime'] == 'low_volatility':
                    if (current_row['close'] < current_row[f'BB_LOWER_{short_window}'] and 
                        current_row['close'] > current_row['ma_200'] and 
                        current_row['RSI_14'] < rsi_oversold and
                        current_funding_rate < funding_rate_threshold):
                        position = 1
                        signal_generated = True
                    elif (current_row['close'] > current_row[f'BB_UPPER_{short_window}'] and
                          current_row['close'] < current_row['ma_200'] and 
                          current_row['RSI_14'] > rsi_overbought and
                          current_funding_rate > -funding_rate_threshold):
                        position = -1
                        signal_generated = True
                
                elif current_row['regime'] == 'high_volatility':
                    if (prev_row[f'EMA_{short_window}'] < prev_row[f'EMA_{long_window}'] and 
                        current_row[f'EMA_{short_window}'] > current_row[f'EMA_{long_window}'] and
                        current_row['close'] > current_row['ma_200'] and
                        current_row['ATR_14'] > atr_threshold.iloc[i] and
                        current_funding_rate < funding_rate_threshold):
                        position = 1
                        signal_generated = True
                    elif (prev_row[f'EMA_{short_window}'] > prev_row[f'EMA_{long_window}'] and 
                          current_row[f'EMA_{short_window}'] < current_row[f'EMA_{long_window}'] and
                          current_row['close'] < current_row['ma_200'] and
                          current_row['ATR_14'] > atr_threshold.iloc[i] and
                          current_funding_rate > -funding_rate_threshold):
                        position = -1
                        signal_generated = True
                
                if signal_generated:
                    entry_price = current_row['open']
                    garch_returns = df['log_returns'].iloc[i-250:i] * 100
                    garch_model = arch_model(garch_returns, vol='Garch', p=1, q=1).fit(disp='off')
                    forecast = garch_model.forecast(horizon=1)
                    estimated_sigma = np.sqrt(forecast.variance.iloc[-1, 0]) / 100
                    
                    stop_loss_pct = estimated_sigma * sl_multiplier
                    take_profit_pct = estimated_sigma * tp_multiplier
                    
                    if position == 1:
                        stop_loss_price = entry_price * (1 - stop_loss_pct)
                        take_profit_price = entry_price * (1 + take_profit_pct)
                    else:
                        stop_loss_price = entry_price * (1 + stop_loss_pct)
                        take_profit_price = entry_price * (1 - take_profit_pct)
    return pnl

param_space = [
    Real(1.0, 3.0, name='sl_multiplier'),
    Real(1.5, 5.0, name='tp_multiplier'),
    Integer(10, 35, name='short_window'),
    Integer(40, 80, name='long_window'),
    Integer(20, 40, name='rsi_oversold'),
    Integer(60, 80, name='rsi_overbought'),
    Real(0.0001, 0.001, name='funding_rate_threshold'),
]

if __name__ == '__main__':
    start_date = '2025-02-20'
    end_date = '2025-08-20'

    print(f"Fetching 6 months of market data ({start_date} to {end_date})...")
    market_data = fetch_historical_data(start_date_str=start_date, end_date_str=end_date)
    
    print("Fetching funding rate data for the same period...")
    # Ensure the 'since' is timezone-aware
    funding_rates = fetch_funding_rates(since=market_data.index[0], timeframe='5m')
    
    # Use merge_asof for robust, time-based joining
    market_data_sorted = market_data.sort_index()
    funding_rates_sorted = funding_rates.sort_index()
    market_data = pd.merge_asof(
        left=market_data_sorted,
        right=funding_rates_sorted,
        left_index=True,
        right_index=True,
        direction='forward'
    )
    
    @use_named_args(param_space)
    def objective(**params):
        pnl = run_full_system_backtest(data=market_data.copy(), **params)
        return -pnl

    print("\nRunning definitive Bayesian Optimization...")
    print("This will be very slow and will take several hours or more.")
    
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
    print("         DEFINITIVE OPTIMIZATION COMPLETE")
    print("="*50)
    print(f"Best PNL Found:      ${best_pnl:.4f}")
    print("\nOptimal Parameters:")
    for param, value in zip(param_space, best_params):
        print(f"- {param.name}: {value}")
    print("="*50)