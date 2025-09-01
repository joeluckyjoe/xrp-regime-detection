import pandas as pd
import numpy as np
import warnings
import pandas_ta as ta
from arch import arch_model
from sklearn.preprocessing import StandardScaler
from pygam import LinearGAM, s
import yfinance as yf

warnings.filterwarnings("ignore")

def fetch_yfinance_data(ticker, start_date, end_date):
    """Fetches historical daily data from Yahoo Finance."""
    df = yf.download(ticker, start=start_date, end=end_date, interval='1d')
    if df.empty:
        return df
    df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
    df = df.tz_localize('UTC')
    return df

def fetch_vix_data(start_date, end_date):
    """Fetches historical VIX data from Yahoo Finance."""
    vix_df = yf.download('^VIX', start=start_date, end=end_date, interval='1d')
    if vix_df.empty:
        return vix_df
    vix_df = vix_df.tz_localize('UTC')
    return vix_df

def run_full_system_backtest(data, sl_multiplier, tp_multiplier, short_window, long_window, rsi_oversold, rsi_overbought, vix_complacency_threshold, transaction_cost=0.001):
    """The full backtesting function, adapted for daily ETF data."""
    df = data.copy()
    log_volume = np.log(df['volume'].replace(0, 1)).dropna()
    if log_volume.empty:
        return 0

    scaler_gam = StandardScaler()
    scaled_log_volume = scaler_gam.fit_transform(log_volume.values.reshape(-1, 1))
    time_idx = np.arange(len(scaled_log_volume))
    
    day_of_week = pd.to_datetime(df.loc[log_volume.index].index).dayofweek
    gam = LinearGAM(s(0, n_splines=4, basis='cp')).fit(day_of_week, scaled_log_volume)
    predictions = gam.predict(day_of_week)
    cycle_mean = predictions.mean()
    
    df = df.loc[log_volume.index].copy()
    df['regime'] = np.where(predictions > cycle_mean, 'high_volatility', 'low_volatility')

    df['ma_200'] = df['close'].rolling(window=200).mean()
    df.ta.rsi(length=14, append=True)
    df.ta.bbands(length=int(short_window), append=True, col_names=(f'BB_LOWER_{int(short_window)}', f'BB_MIDDLE_{int(short_window)}', f'BB_UPPER_{int(short_window)}', 'BB_BANDWIDTH', 'BB_PERCENT'))
    df.ta.ema(length=int(short_window), append=True, col_names=f'EMA_{int(short_window)}')
    df.ta.ema(length=int(long_window), append=True, col_names=f'EMA_{int(long_window)}')
    df.ta.atr(length=14, append=True)
    
    try:
        atr_col_name = [col for col in df.columns if 'ATR' in col][0]
    except IndexError:
        return 0 
    
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df.dropna(inplace=True)
    
    if len(df) < 250:
        return 0

    atr_threshold = df[atr_col_name].rolling(window=20).mean()
    position, pnl, entry_price, stop_loss_price, take_profit_price = 0, 0, 0, 0, 0
    
    for i in range(50, len(df)):
        current_row, prev_row = df.iloc[i], df.iloc[i-1]
        
        if position != 0:
            exit_price = 0
            if position == 1 and (current_row['low'] <= stop_loss_price or current_row['high'] >= take_profit_price):
                exit_price = stop_loss_price if current_row['low'] <= stop_loss_price else take_profit_price
            elif position == -1 and (current_row['high'] >= stop_loss_price or current_row['low'] <= take_profit_price):
                exit_price = stop_loss_price if current_row['high'] >= stop_loss_price else take_profit_price
            if exit_price != 0:
                pnl += (exit_price - entry_price) * position - (abs(exit_price) + abs(entry_price)) * transaction_cost
                position = 0
        
        if position == 0:
            signal_generated = False
            current_atr_threshold, current_vix = atr_threshold.get(current_row.name), current_row.get('vix_close')

            if pd.notna(current_atr_threshold) and pd.notna(current_vix):
                if current_row['regime'] == 'low_volatility':
                    if (current_row['close'] < current_row[f'BB_LOWER_{int(short_window)}'] and current_row['close'] > current_row['ma_200'] and current_row['RSI_14'] < rsi_oversold and current_vix > vix_complacency_threshold):
                        position, signal_generated = 1, True
                    elif (current_row['close'] > current_row[f'BB_UPPER_{int(short_window)}'] and current_row['close'] < current_row['ma_200'] and current_row['RSI_14'] > rsi_overbought and current_vix < vix_complacency_threshold):
                        position, signal_generated = -1, True
                elif current_row['regime'] == 'high_volatility':
                    if (prev_row[f'EMA_{int(short_window)}'] < prev_row[f'EMA_{int(long_window)}'] and current_row[f'EMA_{int(short_window)}'] > current_row[f'EMA_{int(long_window)}'] and current_row['close'] > current_row['ma_200'] and current_row[atr_col_name] > current_atr_threshold and current_vix < vix_complacency_threshold):
                        position, signal_generated = 1, True
                    elif (prev_row[f'EMA_{int(short_window)}'] > prev_row[f'EMA_{int(long_window)}'] and current_row[f'EMA_{int(short_window)}'] < current_row[f'EMA_{int(long_window)}'] and current_row['close'] < current_row['ma_200'] and current_row[atr_col_name] > current_atr_threshold and current_vix > vix_complacency_threshold):
                        position, signal_generated = -1, True
                
                if signal_generated:
                    entry_price = current_row['open']
                    garch_returns = df['log_returns'].iloc[i-50:i] * 100
                    if len(garch_returns) > 5:
                        garch_model = arch_model(garch_returns, vol='Garch', p=1, q=1).fit(disp='off')
                        forecast = garch_model.forecast(horizon=1)
                        estimated_sigma = np.sqrt(forecast.variance.iloc[-1, 0]) / 100
                        stop_loss_pct = estimated_sigma * sl_multiplier
                        take_profit_pct = estimated_sigma * tp_multiplier
                        if position == 1:
                            stop_loss_price, take_profit_price = entry_price * (1 - stop_loss_pct), entry_price * (1 + take_profit_pct)
                        else:
                            stop_loss_price, take_profit_price = entry_price * (1 + stop_loss_pct), entry_price * (1 - take_profit_pct)
    return pnl