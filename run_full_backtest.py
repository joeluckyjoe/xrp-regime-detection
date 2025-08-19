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

def run_backtest_with_risk_management(data, stop_loss_pct=0.005, take_profit_pct=0.01, transaction_cost=0.001, short_window=20, long_window=50):
    """
    Runs a backtest with a stop-loss and take-profit.
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
    
    # 3. Simulate Trades with Risk Management
    position = 0
    pnl = 0
    entry_price = 0
    stop_loss_price = 0
    take_profit_price = 0
    trade_log = []
    
    for i in range(1, len(df)):
        # Check for exit conditions if in a trade
        if position == 1: # Long position
            if df['low'][i] <= stop_loss_price:
                exit_price = stop_loss_price
                trade_pnl = (exit_price - entry_price) - (exit_price + entry_price) * transaction_cost
                pnl += trade_pnl
                trade_log.append({'date': df.index[i], 'type': 'STOP-LOSS', 'price': exit_price, 'pnl': trade_pnl})
                position = 0
            elif df['high'][i] >= take_profit_price:
                exit_price = take_profit_price
                trade_pnl = (exit_price - entry_price) - (exit_price + entry_price) * transaction_cost
                pnl += trade_pnl
                trade_log.append({'date': df.index[i], 'type': 'TAKE-PROFIT', 'price': exit_price, 'pnl': trade_pnl})
                position = 0
        elif position == -1: # Short position
            if df['high'][i] >= stop_loss_price:
                exit_price = stop_loss_price
                trade_pnl = (entry_price - exit_price) - (entry_price + exit_price) * transaction_cost
                pnl += trade_pnl
                trade_log.append({'date': df.index[i], 'type': 'STOP-LOSS', 'price': exit_price, 'pnl': trade_pnl})
                position = 0
            elif df['low'][i] <= take_profit_price:
                exit_price = take_profit_price
                trade_pnl = (entry_price - exit_price) - (entry_price + exit_price) * transaction_cost
                pnl += trade_pnl
                trade_log.append({'date': df.index[i], 'type': 'TAKE-PROFIT', 'price': exit_price, 'pnl': trade_pnl})
                position = 0
        
        # Check for entry conditions if not in a trade
        if position == 0:
            if df['regime'][i] == 'low_volatility':
                if df['close'][i-1] < df['bb_lower'][i-1]:
                    position = 1
                    entry_price = df['open'][i]
                    stop_loss_price = entry_price * (1 - stop_loss_pct)
                    take_profit_price = entry_price * (1 + take_profit_pct)
                    trade_log.append({'date': df.index[i], 'type': 'BUY', 'price': entry_price, 'pnl': np.nan})
                elif df['close'][i-1] > df['bb_upper'][i-1]:
                    position = -1
                    entry_price = df['open'][i]
                    stop_loss_price = entry_price * (1 + stop_loss_pct)
                    take_profit_price = entry_price * (1 - take_profit_pct)
                    trade_log.append({'date': df.index[i], 'type': 'SELL', 'price': entry_price, 'pnl': np.nan})
            elif df['regime'][i] == 'high_volatility':
                if (df['ma_short'][i-2] < df['ma_long'][i-2]) and (df['ma_short'][i-1] > df['ma_long'][i-1]):
                    position = 1
                    entry_price = df['open'][i]
                    stop_loss_price = entry_price * (1 - stop_loss_pct)
                    take_profit_price = entry_price * (1 + take_profit_pct)
                    trade_log.append({'date': df.index[i], 'type': 'BUY', 'price': entry_price, 'pnl': np.nan})
                elif (df['ma_short'][i-2] > df['ma_long'][i-2]) and (df['ma_short'][i-1] < df['ma_long'][i-1]):
                    position = -1
                    entry_price = df['open'][i]
                    stop_loss_price = entry_price * (1 + stop_loss_pct)
                    take_profit_price = entry_price * (1 - take_profit_pct)
                    trade_log.append({'date': df.index[i], 'type': 'SELL', 'price': entry_price, 'pnl': np.nan})

    return pnl, pd.DataFrame(trade_log)

if __name__ == '__main__':
    print("Fetching latest 5-day market data for backtest...")
    market_data = fetch_recent_data(hours_back=120, timeframe='5m')
    
    print("\nRunning quantitative backtest with risk management...")
    # We use a 0.5% stop-loss and a 1% take-profit for this example
    total_pnl, trades = run_backtest_with_risk_management(market_data, stop_loss_pct=0.005, take_profit_pct=0.01)
    
    print("\n" + "="*50)
    print("     QUANTITATIVE BACKTEST (WITH RISK MANAGEMENT)")
    print("="*50)
    print(f"Period Analyzed:     {market_data.index[0]} to {market_data.index[-1]}")
    print(f"Total Trades Taken:  {len(trades[trades['pnl'].notna()])}")
    print(f"Winning Trades:      {len(trades[trades['pnl'] > 0])}")
    print(f"Losing Trades:       {len(trades[trades['pnl'] < 0])}")
    print(f"Total PNL (USD):     ${total_pnl:.4f} (per 1 unit of XRP)")
    print("="*50)

    if len(trades) > 0:
        print("\nLast 10 Trades:")
        print(trades.tail(10))