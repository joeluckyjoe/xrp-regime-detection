import ccxt
import pandas as pd
from datetime import datetime, timedelta

# You can add this as a new function or modify the existing one
def fetch_recent_data(hours_back=24, timeframe='5m', exchange_name='binance'):
    """Fetches recent historical OHLCV data."""
    exchange_class = getattr(ccxt, exchange_name)
    exchange = exchange_class()
    
    since_datetime = datetime.utcnow() - timedelta(hours=hours_back)
    since_timestamp = int(since_datetime.timestamp() * 1000)
    
    # Fetch data in one call (24 hours * 12 5-min intervals = 288 data points, well within the limit)
    ohlcv = exchange.fetch_ohlcv('XRP/USDT', timeframe, since=since_timestamp, limit=1000)

    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def fetch_xrp_usdt_data(days_back=90, timeframe='1h', exchange_name='binance'):
    """
    Fetches historical OHLCV data for XRP/USDT from a specified exchange.
    
    Args:
        days_back (int): How many days of historical data to fetch.
        timeframe (str): The timeframe for the data (e.g., '1h', '4h', '1d').
        exchange_name (str): The name of the exchange to fetch data from.

    Returns:
        pandas.DataFrame: A DataFrame with the historical data.
    """
    exchange_class = getattr(ccxt, exchange_name)
    exchange = exchange_class()
    
    since_datetime = datetime.utcnow() - timedelta(days=days_back)
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

if __name__ == '__main__':
    # Fetch the last 90 days of 1-hour data
    #xrp_data = fetch_xrp_usdt_data(days_back=90, timeframe='1h')
    xrp_data = fetch_recent_data(hours_back=24, timeframe='5m')

    # Save the data to a CSV file
    xrp_data.to_csv('data/xrp_usdt_data.csv')
    print(f"Data for {len(xrp_data)} points fetched and saved to data/xrp_usdt_data.csv")