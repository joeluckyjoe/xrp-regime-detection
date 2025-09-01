import pandas as pd
import pymc as pm
import numpy as np
import arviz as az
import warnings
from sklearn.preprocessing import StandardScaler
import yfinance as yf
from datetime import datetime, timedelta
import traceback

warnings.filterwarnings("ignore")

def fetch_yfinance_data(ticker, start_date, end_date):
    """
    Fetches and cleans the historical OHLC data, handling yfinance's
    potential MultiIndex column format.
    """
    df = yf.download(ticker, start=start_date, end=end_date, interval='1d')
    if df.empty: return df

    # --- THE FIX: Flatten the column index ---
    # yfinance can return a multi-index column; we need to flatten it to single names.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # --- END FIX ---

    # Now we can safely rename the columns
    df.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    }, inplace=True)

    # Ensure all columns are lowercase for consistency
    df.columns = df.columns.str.lower()

    if df.index.tz is None:
        df = df.tz_localize('UTC')
    return df


def run_bayesian_backtest(trace, features_df, price_data, buy_threshold=0.55, sell_threshold=0.55, transaction_cost=0.001):
    """
    Runs a backtest using the trained Bayesian model to generate signals.
    """
    print("Preparing data for backtest...")

    scaler = StandardScaler()
    X = scaler.fit_transform(features_df)

    print("Extracting learned model coefficients...")
    betas = az.extract(trace, var_names=["betas"]).values
    intercept = az.extract(trace, var_names=["intercept"]).values

    print("Generating probabilistic predictions for the entire dataset...")
    all_logits = np.dot(X, betas) + intercept
    all_probabilities = 1 / (1 + np.exp(-all_logits))
    mean_probabilities = all_probabilities.mean(axis=1)

    position = 0
    pnl = 0
    trade_log = []

    print("Starting trade simulation...")
    for i in range(len(mean_probabilities) - 1):
        prob_up = mean_probabilities[i]

        if position != 0:
            exit_price_row = price_data.iloc[i + 1]
            exit_price = exit_price_row['open']

            if pd.isna(exit_price):
                position = 0
                continue

            entry_price = trade_log[-1]['price']
            trade_pnl = (exit_price - entry_price) * position - (abs(exit_price) + abs(entry_price)) * transaction_cost
            pnl += trade_pnl

            for log in reversed(trade_log):
                if pd.isna(log['pnl']):
                    log['pnl'] = trade_pnl
                    break
            position = 0

        entry_price_row = price_data.iloc[i + 1]
        entry_price = entry_price_row['open']

        if pd.isna(entry_price):
            continue

        if prob_up > buy_threshold:
            position = 1
            trade_log.append({'date': entry_price_row.name, 'type': 'BUY', 'price': entry_price, 'pnl': np.nan})
        elif (1 - prob_up) > sell_threshold:
            position = -1
            trade_log.append({'date': entry_price_row.name, 'type': 'SELL', 'price': entry_price, 'pnl': np.nan})

    return pnl, pd.DataFrame(trade_log)

if __name__ == '__main__':
    print("Loading trained model and data...")
    try:
        trace = az.from_netcdf("data/bayesian_signal_model.nc")
        features = pd.read_csv('data/nasdaq_features.csv', index_col=0, parse_dates=True)

        if features.index.tz is None:
            features = features.tz_localize('UTC')

        end_date = datetime.now()
        start_date = end_date - timedelta(days=5*365)
        start_str, end_str = start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')

        print("Fetching full price data for accurate PNL calculation...")
        price_data = fetch_yfinance_data("QQQ", start_str, end_str)

        if not price_data.empty and not features.empty:
            price_data = price_data[~price_data.index.duplicated(keep='first')]
            features = features[~features.index.duplicated(keep='first')]

            common_index = price_data.index.intersection(features.index)

            price_data = price_data.loc[common_index]
            features = features.loc[common_index]

            print(f"Data aligned. Shape of features: {features.shape}, Shape of price_data: {price_data.shape}")

            total_pnl, trades = run_bayesian_backtest(trace, features, price_data)

            print("\n" + "="*50)
            print("              BAYESIAN STRATEGY BACKTEST RESULTS")
            print("="*50)

            if not trades.empty:
                closed_trades = trades.dropna(subset=['pnl'])

                if not closed_trades.empty:
                    closed_trades = closed_trades.copy()
                    closed_trades['pnl'] = pd.to_numeric(closed_trades['pnl'])

                print(f"Total Closed Trades: {len(closed_trades)}")
                if not closed_trades.empty:
                    winning_trades = closed_trades[closed_trades['pnl'] > 0]
                    losing_trades = closed_trades[closed_trades['pnl'] <= 0]

                    print(f"Winning Trades:      {len(winning_trades)}")
                    print(f"Losing Trades:       {len(losing_trades)}")

                    win_rate = len(winning_trades) / len(closed_trades) if len(closed_trades) > 0 else 0
                    print(f"Win Rate:            {win_rate:.2%}")
            else:
                print("Total Trades:        0")

            print(f"\nTotal PNL (USD):     ${total_pnl:.2f} (per 1 unit of QQQ)")
            print("="*50)

            if not trades.dropna(subset=['pnl']).empty:
                print("\nLast 10 Closed Trades:")
                print(trades.dropna(subset=['pnl']).tail(10))
        else:
            print("Could not fetch price data or features data for the backtest.")

    except FileNotFoundError:
        print("Could not find model or data files. Please run the training scripts first.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()