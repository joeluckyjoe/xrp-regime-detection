import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
from tqdm import tqdm
import warnings

# --- Installation required: pip install hurst ---
from hurst import compute_Hc

warnings.filterwarnings("ignore", category=FutureWarning)

# --- CONFIGURATION ---
# A diverse universe of highly liquid ETFs to screen
ASSET_UNIVERSE = [
    # Major US Indices
    "SPY", "QQQ", "IWM", "DIA",
    # Sectors
    "XLF", "XLK", "XLE", "XLV", "XLI",
    # Commodities
    "GLD", "SLV", "USO", "UNG",
    # Fixed Income (Bonds)
    "TLT", "IEF", "HYG",
    # Real Estate & Volatility
    "VNQ", "VXX",
    # International & Emerging Markets
    "EEM", "EFA", "FXI",
    # Currencies & Crypto
    "UUP", "EUO", "FXY", "GBTC"
]
DATA_PERIOD = "10y"

def screen_assets(universe, period):
    """
    Downloads historical data for a universe of assets and calculates metrics
    to gauge their suitability for a trend-following strategy.
    """
    print(f"--- Screening {len(universe)} Assets for Trend-Following Suitability ---")
    print(f"--- Data Period: Last {period} ---")
    
    results = []

    for ticker in tqdm(universe, desc="Analyzing Assets"):
        try:
            # 1. Fetch Data
            data = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
            if data.empty or len(data) < 500: # Need enough data for meaningful stats
                continue

            # 2. Calculate Metrics
            
            # Hurst Exponent: Measures long-term memory. > 0.5 suggests trending.
            hurst_exp, _, _ = compute_Hc(data['Close'], kind='price', simplified=True)

            # Average ADX: Measures trend strength. Higher is better.
            adx_series = ta.adx(data['High'], data['Low'], data['Close'], length=50)
            avg_adx = adx_series['ADX_50'].mean()

            # Autocorrelation: Measures momentum. Positive is better.
            daily_returns = data['Close'].pct_change()
            autocorr = daily_returns.autocorr(lag=1)

            results.append({
                "Ticker": ticker,
                "Hurst": hurst_exp,
                "Avg_ADX_50": avg_adx,
                "Autocorr_1D": autocorr
            })

        except Exception as e:
            # print(f"\nCould not process {ticker}: {e}")
            pass
            
    if not results:
        print("Could not retrieve data for any assets.")
        return

    # 3. Rank the Assets
    results_df = pd.DataFrame(results).set_index("Ticker")
    
    # Create ranks for each metric (higher is better)
    results_df['Hurst_Rank'] = results_df['Hurst'].rank(ascending=False)
    results_df['ADX_Rank'] = results_df['Avg_ADX_50'].rank(ascending=False)
    results_df['Autocorr_Rank'] = results_df['Autocorr_1D'].rank(ascending=False)
    
    # Final score is the sum of ranks (lower sum is better)
    results_df['Total_Rank_Score'] = results_df['Hurst_Rank'] + results_df['ADX_Rank'] + results_df['Autocorr_Rank']
    
    # Sort by the final score
    final_ranking = results_df.sort_values('Total_Rank_Score', ascending=True)

    # 4. Print Report
    print("\n\n--- Asset Suitability Report ---")
    print("Assets are ranked by 'Total_Rank_Score' (lower is better).")
    print("Ideal candidates for our trend-following agent have a high Hurst (>0.5), high Avg_ADX, and positive Autocorrelation.\n")
    
    # Display the most relevant columns
    display_cols = ['Hurst', 'Avg_ADX_50', 'Autocorr_1D', 'Total_Rank_Score']
    print(final_ranking[display_cols].to_string(formatters={
        'Hurst': '{:,.3f}'.format,
        'Avg_ADX_50': '{:,.2f}'.format,
        'Autocorr_1D': '{:,.3f}'.format,
        'Total_Rank_Score': '{:,.0f}'.format
    }))


if __name__ == '__main__':
    screen_assets(ASSET_UNIVERSE, DATA_PERIOD)