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
    """Fetches recent historical OHLCV data, looping to get all data."""
    exchange_class = getattr(ccxt, exchange_name)
    exchange = exchange_class()
    since_datetime = datetime.now(timezone.utc) - timedelta(hours=hours_back)
    since_timestamp = int(since_datetime.timestamp() * 1000)
    
    all_ohlcv = []
    
    while True:
        # Fetch data in batches
        ohlcv = exchange.fetch_ohlcv('XRP/USDT', timeframe, since=since_timestamp, limit=1000)
        if len(ohlcv) == 0:
            break
        all_ohlcv.extend(ohlcv)
        # Set the starting point for the next batch to be after the last candle
        since_timestamp = ohlcv[-1][0] + 1

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def run_final_backtest(data, stop_loss_pct=0.001, take_profit_pct=0.026409605642408934, transaction_cost=0.001, short_window=27, long_window=80):
    """
    Runs a backtest with trend, momentum, and volatility filters.
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
    
    # Define a dynamic ATR threshold (e.g., ATR must be above its own 1-day average)
    atr_threshold = df['ATR_14'].rolling(window=288).mean()

    # 3. Simulate Trades
    position = 0
    pnl = 0
    entry_price = 0
    stop_loss_price = 0
    take_profit_price = 0
    trade_log = []
    
    for i in range(1, len(df)):
        current_row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        # Exit logic
        if position == 1:
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
        elif position == -1:
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
        
        # Entry logic
        if position == 0:
            if current_row['regime'] == 'low_volatility':
                if (current_row['close'] < current_row[f'BB_LOWER_{short_window}'] and 
                    current_row['close'] > current_row['ma_200'] and 
                    current_row['RSI_14'] < 20):
                    position = 1
                    entry_price = current_row['open']
                    stop_loss_price = entry_price * (1 - stop_loss_pct)
                    take_profit_price = entry_price * (1 + take_profit_pct)
                    trade_log.append({'date': current_row.name, 'type': 'BUY', 'price': entry_price, 'pnl': np.nan})
                elif (current_row['close'] > current_row[f'BB_UPPER_{short_window}'] and
                      current_row['close'] < current_row['ma_200'] and 
                      current_row['RSI_14'] > 79):
                    position = -1
                    entry_price = current_row['open']
                    stop_loss_price = entry_price * (1 + stop_loss_pct)
                    take_profit_price = entry_price * (1 - take_profit_pct)
                    trade_log.append({'date': current_row.name, 'type': 'SELL', 'price': entry_price, 'pnl': np.nan})
            
            elif current_row['regime'] == 'high_volatility':
                if (prev_row[f'EMA_{short_window}'] < prev_row[f'EMA_{long_window}'] and 
                    current_row[f'EMA_{short_window}'] > current_row[f'EMA_{long_window}'] and
                    current_row['close'] > current_row['ma_200'] and
                    current_row['ATR_14'] > atr_threshold.iloc[i]):
                    position = 1
                    entry_price = current_row['open']
                    stop_loss_price = entry_price * (1 - stop_loss_pct)
                    take_profit_price = entry_price * (1 + take_profit_pct)
                    trade_log.append({'date': current_row.name, 'type': 'BUY', 'price': entry_price, 'pnl': np.nan})
                elif (prev_row[f'EMA_{short_window}'] > prev_row[f'EMA_{long_window}'] and 
                      current_row[f'EMA_{short_window}'] < current_row[f'EMA_{long_window}'] and
                      current_row['close'] < current_row['ma_200'] and
                      current_row['ATR_14'] > atr_threshold.iloc[i]):
                    position = -1
                    entry_price = current_row['open']
                    stop_loss_price = entry_price * (1 + stop_loss_pct)
                    take_profit_price = entry_price * (1 - take_profit_pct)
                    trade_log.append({'date': current_row.name, 'type': 'SELL', 'price': entry_price, 'pnl': np.nan})

    return pnl, pd.DataFrame(trade_log)

if __name__ == '__main__':
    print("Fetching latest 5-day market data for backtest...")
    market_data = fetch_recent_data(hours_back=360, timeframe='5m')
    
    print("\nRunning final backtest with all filters...")
    total_pnl, trades = run_final_backtest(market_data, stop_loss_pct=0.005, take_profit_pct=0.01)
    
    print("\n" + "="*50)
    print("     QUANTITATIVE BACKTEST (FINAL STRATEGY)")
    print("="*50)
    print(f"Period Analyzed:     {market_data.index[0]} to {market_data.index[-1]}")
    
    # Check if any trades were taken before printing stats
    if not trades.empty and 'pnl' in trades.columns:
        closed_trades = trades[trades['pnl'].notna()]
        print(f"Total Trades Taken:  {len(closed_trades)}")
        if not closed_trades.empty:
            print(f"Winning Trades:      {len(closed_trades[closed_trades['pnl'] > 0])}")
            print(f"Losing Trades:       {len(closed_trades[closed_trades['pnl'] < 0])}")
            win_rate = len(closed_trades[closed_trades['pnl'] > 0]) / len(closed_trades)
            print(f"Win Rate:            {win_rate:.2%}")
    else:
        print("Total Trades Taken:  0")

    print(f"Total PNL (USD):     ${total_pnl:.4f} (per 1 unit of XRP)")
    print("="*50)

    if not trades.empty:
        print("\nLast 10 Trades:")
        print(trades.tail(10))