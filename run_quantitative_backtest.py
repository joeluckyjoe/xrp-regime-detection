import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pygam import LinearGAM, s
from datetime import datetime, timedelta
import ccxt
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def fetch_recent_data(hours_back=120, timeframe='5m', exchange_name='binance'):
    """Fetches recent historical OHLCV data."""
    exchange_class = getattr(ccxt, exchange_name)
    exchange = exchange_class()
    since_datetime = datetime.utcnow() - timedelta(hours=hours_back)
    since_timestamp = int(since_datetime.timestamp() * 1000)
    ohlcv = exchange.fetch_ohlcv('XRP/USDT', timeframe, since=since_timestamp, limit=2000)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def run_backtest(data, short_window=20, long_window=50, transaction_cost=0.001):
    """
    Runs a full backtest of the dynamic strategy and calculates profit/loss.
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

    # 2. Calculate indicators
    df['bb_middle'] = df['close'].rolling(window=short_window).mean()
    df['bb_std'] = df['close'].rolling(window=short_window).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
    df['ma_short'] = df['close'].rolling(window=short_window).mean()
    df['ma_long'] = df['close'].rolling(window=long_window).mean()
    
    # 3. Simulate Trades
    position = 0  # -1 for short, 0 for flat, 1 for long
    pnl = 0
    trade_log = []
    
    for i in range(1, len(df)):
        # Exit conditions (simplified: exit at end of bar)
        if position != 0:
            entry_price = trade_log[-1]['price']
            exit_price = df['open'][i]
            trade_pnl = (exit_price - entry_price) * position - (abs(exit_price) + abs(entry_price)) * transaction_cost
            pnl += trade_pnl
            trade_log[-1]['pnl'] = trade_pnl
            position = 0
        
        # Entry conditions
        if df['regime'][i] == 'low_volatility':
            if df['close'][i-1] < df['bb_lower'][i-1]:
                position = 1 # Buy
                trade_log.append({'date': df.index[i], 'type': 'BUY', 'price': df['open'][i]})
            elif df['close'][i-1] > df['bb_upper'][i-1]:
                position = -1 # Sell
                trade_log.append({'date': df.index[i], 'type': 'SELL', 'price': df['open'][i]})
                
        elif df['regime'][i] == 'high_volatility':
            if (df['ma_short'][i-2] < df['ma_long'][i-2]) and (df['ma_short'][i-1] > df['ma_long'][i-1]):
                position = 1 # Buy
                trade_log.append({'date': df.index[i], 'type': 'BUY', 'price': df['open'][i]})
            elif (df['ma_short'][i-2] > df['ma_long'][i-2]) and (df['ma_short'][i-1] < df['ma_long'][i-1]):
                position = -1 # Sell
                trade_log.append({'date': df.index[i], 'type': 'SELL', 'price': df['open'][i]})

    return pnl, pd.DataFrame(trade_log)

if __name__ == '__main__':
    print("Fetching latest 5-day market data for backtest...")
    market_data = fetch_recent_data(hours_back=120, timeframe='5m')
    
    print("\nRunning quantitative backtest...")
    total_pnl, trades = run_backtest(market_data)
    
    print("\n" + "="*50)
    print("           QUANTITATIVE BACKTEST RESULTS")
    print("="*50)
    print(f"Period Analyzed:     {market_data.index[0]} to {market_data.index[-1]}")
    print(f"Total Trades:        {len(trades)}")
    print(f"Total PNL (USD):     ${total_pnl:.4f} (per 1 unit of XRP)")
    print("="*50)
    
    if len(trades) > 0:
        print("\nLast 5 Trades:")
        print(trades.tail())