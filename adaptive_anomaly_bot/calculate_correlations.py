import pandas as pd
import yfinance as yf
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)

# --- CONFIGURATION ---
# The top 10 assets identified in our previous trend-screening experiment
TOP_ASSETS = [
    "GBTC", "GLD", "USO", "HYG", "VXX", 
    "IEF", "UNG", "UUP", "QQQ", "XLF"
]
DATA_PERIOD = "10y"

def calculate_and_plot_correlations(universe, period):
    """
    Downloads historical data for a universe of assets, calculates their
    return correlations, and plots a heatmap.
    """
    print(f"--- Calculating Correlation Matrix for Top {len(universe)} Assets ---")
    
    all_returns = {}

    for ticker in tqdm(universe, desc="Fetching Data"):
        try:
            data = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
            if data.empty or len(data) < 500:
                print(f"\nSkipping {ticker} due to insufficient data.")
                continue
            
            # Calculate daily returns and store them
            all_returns[ticker] = data['Close'].pct_change()

        except Exception as e:
            print(f"\nCould not process {ticker}: {e}")
            pass
            
    if not all_returns:
        print("Could not retrieve data for any assets.")
        return

    # Create a single DataFrame with all asset returns, aligning by date
    returns_df = pd.DataFrame(all_returns).dropna()
    
    # Calculate the correlation matrix
    correlation_matrix = returns_df.corr()
    
    # --- Print and Plot Results ---
    
    # Print the raw matrix
    print("\n\n--- Correlation Matrix of Daily Returns ---")
    print(correlation_matrix.to_string(float_format='{:,.2f}'.format))
    
    # Create the heatmap
    print("\nGenerating heatmap...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap='coolwarm',
        fmt=".2f",
        linewidths=.5,
        ax=ax
    )
    
    ax.set_title('Correlation Matrix of Top Trending Assets', fontsize=16)
    plot_filename = 'correlation_heatmap.png'
    plt.savefig(plot_filename)
    print(f"Heatmap saved successfully as '{plot_filename}'")
    
    print("Displaying plot...")
    plt.show()


if __name__ == '__main__':
    calculate_and_plot_correlations(TOP_ASSETS, DATA_PERIOD)