import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from src.portfolio_engine import fetch_yfinance_data, fetch_vix_data, run_full_system_backtest

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
ASSETS = {
    "NASDAQ": "QQQ",
    "Gold": "GLD",
    "EURUSD": "EURUSD=X"
}
OPTIMIZATION_PERIOD_YEARS = 2

param_space = [
    Real(1.0, 5.0, name='sl_multiplier'), Real(1.5, 8.0, name='tp_multiplier'),
    Integer(10, 50, name='short_window'), Integer(50, 200, name='long_window'),
    Integer(20, 40, name='rsi_oversold'), Integer(60, 80, name='rsi_overbought'),
    Integer(15, 35, name='vix_complacency_threshold'),
]

if __name__ == '__main__':
    end_date = datetime.now()
    start_date = end_date - timedelta(days=OPTIMIZATION_PERIOD_YEARS*365)
    start_date_str, end_date_str = start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')

    print("Fetching VIX data for sentiment filter...")
    vix_data = fetch_vix_data(start_date, end_date)
    
    for name, ticker in ASSETS.items():
        print("\n" + "="*60)
        print(f"STARTING OPTIMIZATION FOR: {name} ({ticker})")
        print("="*60)
        
        print(f"Fetching {OPTIMIZATION_PERIOD_YEARS} years of market data for {name}...")
        market_data = fetch_yfinance_data(ticker, start_date_str, end_date_str)
        
        if market_data.empty:
            print(f"Could not fetch data for {name}. Skipping.")
            continue
            
        market_data = pd.merge_asof(
            left=market_data.sort_index(),
            right=vix_data[['Close']].rename(columns={'Close': 'vix_close'}),
            left_index=True, right_index=True, direction='forward'
        )

        data_is_valid = True
        if 'vix_close' not in market_data.columns:
            data_is_valid = False
        elif market_data['vix_close'].isna().all():
            data_is_valid = False

        if not data_is_valid:
            print(f"Could not merge VIX data for {name}. Skipping.")
            continue
            
        @use_named_args(param_space)
        def objective(**params):
            pnl = run_full_system_backtest(data=market_data.copy(), **params)
            return -pnl

        print(f"Running Bayesian Optimization for {name}...")
        print("This may take a while...")
        result = gp_minimize(objective, param_space, n_calls=50, random_state=0, n_jobs=-1)
        
        best_pnl = -result.fun
        best_params_list = result.x
        best_params_dict = {param.name: value for param, value in zip(param_space, best_params_list)}

        param_filename = f"parameters_{name.lower()}_1d.json"
        with open(param_filename, 'w') as f:
            for key, value in best_params_dict.items():
                if isinstance(value, np.integer): best_params_dict[key] = int(value)
                elif isinstance(value, np.floating): best_params_dict[key] = float(value)
            json.dump(best_params_dict, f, indent=4)
        print(f"\nOptimization for {name} complete. Parameters saved to {param_filename}")
        
        print("\n" + "="*50)
        print(f"     OPTIMIZATION COMPLETE FOR: {name}")
        print("="*50)
        print(f"Best PNL Found:      ${best_pnl:.4f} (per 1 unit)")
        print("\nOptimal Parameters:")
        for p_name, value in best_params_dict.items():
            print(f"- {p_name}: {value}")
        print("="*50)