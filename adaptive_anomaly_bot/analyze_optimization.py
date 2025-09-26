import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze_and_plot(log_filename="robust_optimization_log_TLT_LCB.csv"):
    """
    Loads the optimization log, analyzes the results, prints the best
    parameters, and generates a plot.
    """
    try:
        log_df = pd.read_csv(log_filename)
        print(f"Successfully loaded '{log_filename}'. Analyzing {len(log_df)} completed runs...")

        if log_df.empty:
            print("The log file is empty. No data to analyze.")
            return

        # --- Analysis ---
        best_lcb_run = log_df.loc[log_df['lcb_score'].idxmax()]
        most_stable_run = log_df.loc[log_df['std_dev_score'].idxmin()]

        print("\n--- Best Overall Run So Far (Highest LCB Score) ---")
        print(f"LCB Score: ${best_lcb_run['lcb_score']:,.2f}")
        print(f"Mean Return: ${best_lcb_run['mean_score']:,.2f}")
        print(f"Standard Deviation: ${best_lcb_run['std_dev_score']:,.2f}")

        # +++ ADDED SECTION TO PRINT BEST PARAMETERS +++
        print("\n--- Best Parameters Found (Highest LCB Score) ---")
        print(f"  - BULL:     lr={best_lcb_run['lr_bull']:.6f}, entropy_coef={best_lcb_run['entropy_coef_bull']:.4f}")
        print(f"  - BEAR:     lr={best_lcb_run['lr_bear']:.6f}, entropy_coef={best_lcb_run['entropy_coef_bear']:.4f}")
        print(f"  - SIDEWAYS: lr={best_lcb_run['lr_sideways']:.6f}, entropy_coef={best_lcb_run['entropy_coef_sideways']:.4f}")
        
        print("\n--- Most Stable Run So Far (Lowest Standard Deviation) ---")
        print(f"Standard Deviation: ${most_stable_run['std_dev_score']:,.2f}")
        print(f"Mean Return: ${most_stable_run['mean_score']:,.2f}")
        print(f"LCB Score: ${most_stable_run['lcb_score']:,.2f}")

        # --- Visualization ---
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 8))

        sc = ax.scatter(log_df['std_dev_score'], log_df['mean_score'], c=log_df['lcb_score'], cmap='viridis', alpha=0.7, label='Tested Parameters')
        ax.scatter(best_lcb_run['std_dev_score'], best_lcb_run['mean_score'], color='red', s=200, edgecolors='black', label=f'Best LCB Score (${best_lcb_run["lcb_score"]:,.0f})', zorder=5)
        ax.scatter(most_stable_run['std_dev_score'], most_stable_run['mean_score'], color='cyan', s=200, edgecolors='black', marker='X', label=f'Most Stable (Std Dev ${most_stable_run["std_dev_score"]:,.0f})', zorder=5)

        ax.set_xlabel("Variability (Standard Deviation)", fontsize=12)
        ax.set_ylabel("Profitability (Mean Return)", fontsize=12)
        ax.set_title(f'Profitability vs. Variability for {len(log_df)} Optimization Runs', fontsize=16)
        
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'${y:,.0f}'))
        
        cbar = plt.colorbar(sc)
        cbar.set_label('LCB Score (Higher is Better)')
        ax.legend()
        fig.tight_layout()

        plot_filename = 'optimization_analysis_plot_HYG.png'
        plt.savefig(plot_filename)
        print(f"\nPlot saved successfully as '{plot_filename}'")
        
        print("Displaying plot...")
        plt.show()

    except FileNotFoundError:
        print(f"ERROR: The log file '{log_filename}' was not found.")
        print("Please make sure this script is in the same directory as your log file.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    # You can change the filename here if you are analyzing a different asset log
    analyze_and_plot(log_filename="robust_optimization_log_HYG_FINAL.csv")