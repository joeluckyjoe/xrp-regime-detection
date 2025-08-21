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

# --- Optimized parameters from your successful 6-month run ---
OPTIMIZED_PARAMS = {
    "sl_multiplier": 3.0,
    "tp_multiplier": 5.0,
    "short_window": 12,
    "long_window": 80,
    "rsi_oversold": 20,
    "rsi_overbought": 76,
    "funding_rate_threshold": 0.0001
}

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
    df = df.tz_localize('UTC')
    return df[(df.index >= start_dt) & (df.index < end_dt)]

def fetch_funding_rates(since, timeframe='5m', exchange_name='binanceusdm'):
    """Fetches historical funding rate data, looping to get all data."""
    exchange = getattr(ccxt, exchange_name)()
    since_timestamp = int(since.timestamp() * 1000)
    all_funding = []
    while True:
        funding_data = exchange.fetch_funding_rate_history('XRP/USDT:USDT', since=since_timestamp, limit=1000)
        if len(funding_data) == 0: break
        all_funding.extend(funding_data)
        since_timestamp = funding_data[-1]['timestamp'] + 1
    df = pd.DataFrame(all_funding)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp')['fundingRate'].astype(float)
    df = df.tz_localize('UTC')
    return df.resample(timeframe).ffill()

def run_full_system_backtest(data, sl_multiplier, tp_multiplier, short_window, long_window, rsi_oversold, rsi_overbought, funding_rate_threshold, transaction_cost=0.001):
    """
    The full backtesting function that returns the DataFrame with signals.
    """
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
    df.ta.bbands(length=int(short_window), append=True, col_names=(f'BB_LOWER_{int(short_window)}', f'BB_MIDDLE_{int(short_window)}', f'BB_UPPER_{int(short_window)}', 'BB_BANDWIDTH', 'BB_PERCENT'))
    df.ta.ema(length=int(short_window), append=True, col_names=f'EMA_{int(short_window)}')
    df.ta.ema(length=int(long_window), append=True, col_names=f'EMA_{int(long_window)}')
    df.ta.atr(length=14, append=True, col_names='ATR_14')
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df.dropna(inplace=True)
    
    atr_threshold = df['ATR_14'].rolling(window=288).mean()

    position = 0
    pnl = 0
    entry_price = 0
    stop_loss_price = 0
    take_profit_price = 0
    trade_log = []
    
    df['bb_buy'] = False
    df['bb_sell'] = False
    df['ma_buy'] = False
    df['ma_sell'] = False
    
    for i in range(250, len(df)):
        current_row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        # Exit logic
        if position != 0:
            exit_price = 0
            trade_type = ''
            if position == 1:
                if current_row['low'] <= stop_loss_price:
                    exit_price, trade_type = stop_loss_price, 'STOP-LOSS'
                elif current_row['high'] >= take_profit_price:
                    exit_price, trade_type = take_profit_price, 'TAKE-PROFIT'
            elif position == -1:
                if current_row['high'] >= stop_loss_price:
                    exit_price, trade_type = stop_loss_price, 'STOP-LOSS'
                elif current_row['low'] <= take_profit_price:
                    exit_price, trade_type = take_profit_price, 'TAKE-PROFIT'
            
            if exit_price != 0:
                trade_pnl = (exit_price - entry_price) * position - (abs(exit_price) + abs(entry_price)) * transaction_cost
                pnl += trade_pnl
                trade_log.append({'date': current_row.name, 'type': trade_type, 'price': exit_price, 'pnl': trade_pnl})
                position = 0
        
        # Entry logic
        if position == 0:
            current_funding_rate = current_row.get('fundingRate')
            if pd.notna(current_funding_rate):
                signal_generated = False
                current_atr_threshold = atr_threshold.get(current_row.name)
                if pd.notna(current_atr_threshold):
                    if current_row['regime'] == 'low_volatility':
                        if (current_row['close'] < current_row[f'BB_LOWER_{int(short_window)}'] and current_row['close'] > current_row['ma_200'] and current_row['RSI_14'] < rsi_oversold and current_funding_rate < funding_rate_threshold):
                            position, signal_generated = 1, True
                            df.loc[current_row.name, 'bb_buy'] = True
                        elif (current_row['close'] > current_row[f'BB_UPPER_{int(short_window)}'] and current_row['close'] < current_row['ma_200'] and current_row['RSI_14'] > rsi_overbought and current_funding_rate > -funding_rate_threshold):
                            position, signal_generated = -1, True
                            df.loc[current_row.name, 'bb_sell'] = True
                    elif current_row['regime'] == 'high_volatility':
                        if (prev_row[f'EMA_{int(short_window)}'] < prev_row[f'EMA_{int(long_window)}'] and current_row[f'EMA_{int(short_window)}'] > current_row[f'EMA_{int(long_window)}'] and current_row['close'] > current_row['ma_200'] and current_row['ATR_14'] > current_atr_threshold and current_funding_rate < funding_rate_threshold):
                            position, signal_generated = 1, True
                            df.loc[current_row.name, 'ma_buy'] = True
                        elif (prev_row[f'EMA_{int(short_window)}'] > prev_row[f'EMA_{int(long_window)}'] and current_row[f'EMA_{int(short_window)}'] < current_row[f'EMA_{long_window}'] and current_row['close'] < current_row['ma_200'] and current_row['ATR_14'] > current_atr_threshold and current_funding_rate > -funding_rate_threshold):
                            position, signal_generated = -1, True
                            df.loc[current_row.name, 'ma_sell'] = True
                
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
                        trade_log.append({'date': current_row.name, 'type': 'BUY', 'price': entry_price, 'pnl': np.nan})
                    else:
                        stop_loss_price = entry_price * (1 + stop_loss_pct)
                        take_profit_price = entry_price * (1 - take_profit_pct)
                        trade_log.append({'date': current_row.name, 'type': 'SELL', 'price': entry_price, 'pnl': np.nan})

    return pnl, pd.DataFrame(trade_log), df

if __name__ == '__main__':
    start_date = '2024-08-20'
    end_date = '2025-02-20'

    print(f"Fetching unseen historical data for validation ({start_date} to {end_date})...")
    validation_data = fetch_historical_data(start_date_str=start_date, end_date_str=end_date)
    print("Fetching funding rate data...")
    funding_rates = fetch_funding_rates(since=validation_data.index[0], timeframe='5m')
    
    validation_data = pd.merge_asof(
        left=validation_data.sort_index(),
        right=funding_rates.sort_index(),
        left_index=True,
        right_index=True,
        direction='forward'
    )
    
    print("\nRunning final validation backtest...")
    total_pnl, trades, backtest_df = run_full_system_backtest(
        validation_data,
        sl_multiplier=OPTIMIZED_PARAMS['sl_multiplier'],
        tp_multiplier=OPTIMIZED_PARAMS['tp_multiplier'],
        short_window=OPTIMIZED_PARAMS['short_window'],
        long_window=OPTIMIZED_PARAMS['long_window'],
        rsi_oversold=OPTIMIZED_PARAMS['rsi_oversold'],
        rsi_overbought=OPTIMIZED_PARAMS['rsi_overbought'],
        funding_rate_threshold=OPTIMIZED_PARAMS['funding_rate_threshold']
    )
    
    # Save the full results DataFrame for plotting in the notebook
    backtest_df.to_csv('data/validation_plot_data.csv')
    print("\nFull backtest data saved to 'data/validation_plot_data.csv'")

    print("\n" + "="*50)
    print("           FINAL VALIDATION RESULTS")
    print("="*50)
    print(f"Period Analyzed:     {backtest_df.index.min()} to {backtest_df.index.max()}")
    if not trades.empty:
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