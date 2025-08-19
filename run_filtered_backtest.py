import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pygam import LinearGAM, s
from datetime import datetime, timedelta, timezone
import ccxt
import warnings
import pandas_ta as ta

# Suppress standard warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def fetch_recent_data(hours_back=120, timeframe='5m', exchange_name='binance'):
    """Fetches recent historical OHLCV data."""
    exchange_class = getattr(ccxt, exchange_name)
    exchange = exchange_class()
    # Use timezone-aware UTC datetime object
    since_datetime = datetime.now(timezone.utc) - timedelta(hours=hours_back)
    since_timestamp = int(since_datetime.timestamp() * 1000)
    ohlcv = exchange.fetch_ohlcv('XRP/USDT', timeframe, since=since_timestamp, limit=2000)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def run_filtered_backtest(data, stop_loss_pct=0.005, take_profit_pct=0.01, transaction_cost=0.001, short_window=20, long_window=50):
    """
    Runs a backtest with added trend and momentum filters.
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
    df['bb_middle'] = df['close'].rolling(window=short_window).mean()
    df['bb_std'] = df['close'].rolling(window=short_window).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
    df['ma_short'] = df['close'].rolling(window=short_window).mean()
    df['ma_long'] = df['close'].rolling(window=long_window).mean()
    df['ma_200'] = df['close'].rolling(window=200).mean() # Trend Filter
    df.ta.rsi(length=14, append=True) # Momentum Filter (creates 'RSI_14' column)
    df.dropna(inplace=True) # Drop rows with NaNs from indicator calculations
    
    # 3. Simulate Trades with new, stricter rules
    position = 0
    pnl = 0
    entry_price = 0
    stop_loss_price = 0
    take_profit_price = 0
    trade_log = []
    
    for i in range(len(df)):
        # Use .iloc for safe positional access
        current_row = df.iloc[i]
        
        # Check for exit conditions if in a trade
        if position == 1: # Long position
            if current_row['low'] <= stop_loss_price:
                exit_price = stop_loss_price
                trade_pnl = (exit_price - entry_price) - (exit_price + entry_price) * transaction_cost
                pnl += trade_pnl
                trade_log.append({'date': current_row.name, 'type': 'STOP-LOSS', 'price': exit_price, 'pnl': trade_pnl})
                position = 0
            elif current_row['high'] >= take_profit_price:
                exit_price = take_profit_price
                trade_pnl = (exit_price - entry_price) - (exit_price + entry_price) * transaction_cost
                pnl += trade_pnl
                trade_log.append({'date': current_row.name, 'type': 'TAKE-PROFIT', 'price': exit_price, 'pnl': trade_pnl})
                position = 0
        elif position == -1: # Short position
            if current_row['high'] >= stop_loss_price:
                exit_price = stop_loss_price
                trade_pnl = (entry_price - exit_price) - (entry_price + exit_price) * transaction_cost
                pnl += trade_pnl
                trade_log.append({'date': current_row.name, 'type': 'STOP-LOSS', 'price': exit_price, 'pnl': trade_pnl})
                position = 0
            elif current_row['low'] <= take_profit_price:
                exit_price = take_profit_price
                trade_pnl = (entry_price - exit_price) - (entry_price + exit_price) * transaction_cost
                pnl += trade_pnl
                trade_log.append({'date': current_row.name, 'type': 'TAKE-PROFIT', 'price': exit_price, 'pnl': trade_pnl})
                position = 0
        
        # Check for entry conditions if not in a trade
        if position == 0:
            if current_row['regime'] == 'low_volatility':
                if (current_row['close'] < current_row['bb_lower'] and 
                    current_row['close'] > current_row['ma_200'] and 
                    current_row['RSI_14'] < 30):
                    position = 1
                    entry_price = current_row['open']
                    stop_loss_price = entry_price * (1 - stop_loss_pct)
                    take_profit_price = entry_price * (1 + take_profit_pct)
                    trade_log.append({'date': current_row.name, 'type': 'BUY', 'price': entry_price, 'pnl': np.nan})
                elif (current_row['close'] > current_row['bb_upper'] and
                      current_row['close'] < current_row['ma_200'] and 
                      current_row['RSI_14'] > 70):
                    position = -1
                    entry_price = current_row['open']
                    stop_loss_price = entry_price * (1 + stop_loss_pct)
                    take_profit_price = entry_price * (1 - take_profit_pct)
                    trade_log.append({'date': current_row.name, 'type': 'SELL', 'price': entry_price, 'pnl': np.nan})
            
            elif current_row['regime'] == 'high_volatility':
                prev_row = df.iloc[i-1]
                if (prev_row['ma_short'] < prev_row['ma_long'] and 
                    current_row['ma_short'] > current_row['ma_long'] and
                    current_row['close'] > current_row['ma_200']):
                    position = 1
                    entry_price = current_row['open']
                    stop_loss_price = entry_price * (1 - stop_loss_pct)
                    take_profit_price = entry_price * (1 + take_profit_pct)
                    trade_log.append({'date': current_row.name, 'type': 'BUY', 'price': entry_price, 'pnl': np.nan})
                elif (prev_row['ma_short'] > prev_row['ma_long'] and 
                      current_row['ma_short'] < current_row['ma_long'] and
                      current_row['close'] < current_row['ma_200']):
                    position = -1
                    entry_price = current_row['open']
                    stop_loss_price = entry_price * (1 + stop_loss_pct)
                    take_profit_price = entry_price * (1 - take_profit_pct)
                    trade_log.append({'date': current_row.name, 'type': 'SELL', 'price': entry_price, 'pnl': np.nan})

    return pnl, pd.DataFrame(trade_log)

if __name__ == '__main__':
    print("Fetching latest 5-day market data for backtest...")
    market_data = fetch_recent_data(hours_back=120, timeframe='5m')
    
    print("\nRunning filtered backtest with risk management...")
    total_pnl, trades = run_filtered_backtest(market_data, stop_loss_pct=0.005, take_profit_pct=0.01)
    
    print("\n" + "="*50)
    print("     QUANTITATIVE BACKTEST (WITH FILTERS & RISK MGMT)")
    print("="*50)
    print(f"Period Analyzed:     {market_data.index[0]} to {market_data.index[-1]}")
    print(f"Total Trades Taken:  {len(trades[trades['pnl'].notna()])}")
    if len(trades[trades['pnl'].notna()]) > 0:
        print(f"Winning Trades:      {len(trades[trades['pnl'] > 0])}")
        print(f"Losing Trades:       {len(trades[trades['pnl'] < 0])}")
        win_rate = len(trades[trades['pnl'] > 0]) / len(trades[trades['pnl'].notna()])
        print(f"Win Rate:            {win_rate:.2%}")
    print(f"Total PNL (USD):     ${total_pnl:.4f} (per 1 unit of XRP)")
    print("="*50)

    if len(trades) > 0:
        print("\nLast 10 Trades:")
        print(trades.tail(10))