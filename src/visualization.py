import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import arviz as az

def plot_dynamic_backtest(indicator_df, trace):
    """
    Plots the price with both types of signals, dynamically adjusted
    for the probability of their respective regimes.
    """
    # --- Get Regime Info (same as in the notebook) ---
    sigma_posteriors = az.extract(trace, var_names=["sigma"]).values
    weight_posteriors = az.extract(trace, var_names=["weight"]).values
    mean_sigma_regime_0 = sigma_posteriors[0, :].mean()
    mean_sigma_regime_1 = sigma_posteriors[1, :].mean()

    if mean_sigma_regime_0 < mean_sigma_regime_1:
        low_vol_regime_index = 0
    else:
        low_vol_regime_index = 1
    
    # Get the continuous probability of being in the LOW volatility regime
    if low_vol_regime_index == 0:
        prob_low_vol = 1 - weight_posteriors.mean(axis=1)
    else:
        prob_low_vol = weight_posteriors.mean(axis=1)
        
    prob_high_vol = 1 - prob_low_vol

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(18, 9))
    ax.plot(indicator_df.index, indicator_df['close'], label='Close Price', color='k', alpha=0.6, lw=1)

    # --- Plot Bollinger Signals with dynamic transparency ---
    bb_buy_signals = indicator_df[indicator_df['bb_buy_signal']]
    bb_sell_signals = indicator_df[indicator_df['bb_sell_signal']]
    
    for i in range(len(bb_buy_signals)):
        ax.plot(bb_buy_signals.index[i], bb_buy_signals['close'][i], '^', markersize=10, 
                color='green', alpha=prob_low_vol[bb_buy_signals.index.get_loc(bb_buy_signals.index[i])])
        
    for i in range(len(bb_sell_signals)):
        ax.plot(bb_sell_signals.index[i], bb_sell_signals['close'][i], 'v', markersize=10, 
                color='red', alpha=prob_low_vol[bb_sell_signals.index.get_loc(bb_sell_signals.index[i])])

    # --- Plot Moving Average Signals with dynamic transparency ---
    ma_buy_signals = indicator_df[indicator_df['ma_buy_signal']]
    ma_sell_signals = indicator_df[indicator_df['ma_sell_signal']]
    
    for i in range(len(ma_buy_signals)):
        ax.plot(ma_buy_signals.index[i], ma_buy_signals['close'][i], 'o', markersize=8, 
                color='cyan', alpha=prob_high_vol[ma_buy_signals.index.get_loc(ma_buy_signals.index[i])], markeredgecolor='k')

    for i in range(len(ma_sell_signals)):
        ax.plot(ma_sell_signals.index[i], ma_sell_signals['close'][i], 'o', markersize=8, 
                color='magenta', alpha=prob_high_vol[ma_sell_signals.index.get_loc(ma_sell_signals.index[i])], markeredgecolor='k')

    # Create custom legend entries
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='k', lw=1, label='Close Price'),
                       Line2D([0], [0], marker='^', color='green', label='Mean-Reversion Buy', markersize=10, ls=''),
                       Line2D([0], [0], marker='v', color='red', label='Mean-Reversion Sell', markersize=10, ls=''),
                       Line2D([0], [0], marker='o', color='cyan', label='Momentum Buy', markersize=8, ls='', markeredgecolor='k'),
                       Line2D([0], [0], marker='o', color='magenta', label='Momentum Sell', markersize=8, ls='', markeredgecolor='k')]

    ax.set_title('Dynamic Backtest with Regime-Weighted Signals')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USDT)')
    ax.legend(handles=legend_elements)
    ax.grid(True)
    plt.show()

def plot_price_and_returns(price_data, log_returns):
    """Plots the closing price and log returns."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    ax1.plot(price_data.index, price_data['close'], label='Close Price')
    ax1.set_title('XRP/USDT Close Price')
    ax1.set_ylabel('Price (USDT)')
    ax1.grid(True)
    ax1.legend()

    ax2.plot(log_returns.index, log_returns, label='Log Returns', color='orange')
    ax2.set_title('XRP/USDT Log Returns')
    ax2.set_ylabel('Log Return')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()

def plot_regime_transition(price_data, trace, log_returns):
    """
    Plots the price with the detected smooth regime transition overlaid.
    """
    # Extract the posterior samples for the change point
    change_point_samples = az.extract(trace, var_names=["change_point"]).values
    
    # Calculate the mean change point index
    mean_cp_index = int(np.mean(change_point_samples))
    mean_cp_date = log_returns.index[mean_cp_index]
    
    # Extract the transition weights
    weight_samples = az.extract(trace, var_names=["weight"]).values
    mean_weights = weight_samples.mean(axis=1)

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(15, 7))

    # Plot the price on the primary y-axis
    ax1.plot(price_data.index, price_data['close'], label='Close Price', color='k', alpha=0.7)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price (USDT)', color='k')
    ax1.tick_params(axis='y', labelcolor='k')
    ax1.grid(True, axis='x')
    
    # Add a vertical line for the mean change point
    ax1.axvline(mean_cp_date, color='r', linestyle='--', lw=2, label=f'Mean Change Point ({mean_cp_date.date()})')

    # Create a secondary y-axis for the regime probability
    ax2 = ax1.twinx()
    ax2.plot(log_returns.index, mean_weights, color='C0', lw=2, label='Probability of High-Vol Regime')
    ax2.set_ylabel('Probability of High-Vol Regime', color='C0')
    ax2.tick_params(axis='y', labelcolor='C0')
    ax2.set_ylim(-0.05, 1.05)

    fig.suptitle('XRP/USDT Price and Volatility Regime Transition', fontsize=16)
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_trading_signals(indicator_df, regime_type):
    """
    Plots the price and the appropriate trading signals based on the detected regime.
    """
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(indicator_df.index, indicator_df['close'], label='Close Price', color='k', alpha=0.5)

    if regime_type == 'low_volatility':
        # Plot Bollinger Bands and signals
        ax.plot(indicator_df.index, indicator_df['bb_upper'], 'b--', label='Upper Band')
        ax.plot(indicator_df.index, indicator_df['bb_lower'], 'b--', label='Lower Band')
        
        buy_signals = indicator_df[indicator_df['bb_buy_signal']]
        sell_signals = indicator_df[indicator_df['bb_sell_signal']]
        
        ax.plot(buy_signals.index, buy_signals['close'], '^', markersize=10, color='g', label='Bollinger Buy Signal')
        ax.plot(sell_signals.index, sell_signals['close'], 'v', markersize=10, color='r', label='Bollinger Sell Signal')
        ax.set_title('Trading Signals for Low-Volatility (Mean-Reversion) Regime')

    elif regime_type == 'high_volatility':
        # Plot Moving Averages and signals
        ax.plot(indicator_df.index, indicator_df['ma_short'], 'C0', label='Short MA')
        ax.plot(indicator_df.index, indicator_df['ma_long'], 'C1', label='Long MA')
        
        buy_signals = indicator_df[indicator_df['ma_buy_signal']]
        sell_signals = indicator_df[indicator_df['ma_sell_signal']]
        
        ax.plot(buy_signals.index, buy_signals['close'], '^', markersize=10, color='g', label='MA Crossover Buy Signal')
        ax.plot(sell_signals.index, sell_signals['close'], 'v', markersize=10, color='r', label='MA Crossover Sell Signal')
        ax.set_title('Trading Signals for High-Volatility (Momentum) Regime')

    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USDT)')
    ax.legend()
    ax.grid(True)
    plt.show()

def plot_gam_backtest(backtest_df):
    """
    Plots the price with the active trading signals determined by the GAM volume regime.
    """
    fig, ax = plt.subplots(figsize=(18, 9))
    ax.plot(backtest_df.index, backtest_df['close'], label='Close Price', color='k', alpha=0.4, lw=1)

    # Separate data by regime
    low_vol_data = backtest_df[backtest_df['regime'] == 'low_volatility']
    high_vol_data = backtest_df[backtest_df['regime'] == 'high_volatility']
    
    # Plot Bollinger Band signals for low-vol periods
    bb_buy = low_vol_data[low_vol_data['bb_buy']]
    bb_sell = low_vol_data[low_vol_data['bb_sell']]
    ax.plot(bb_buy.index, bb_buy['close'], '^', markersize=10, color='green', label='Mean-Reversion Buy')
    ax.plot(bb_sell.index, bb_sell['close'], 'v', markersize=10, color='red', label='Mean-Reversion Sell')

    # Plot Moving Average signals for high-vol periods
    ma_buy = high_vol_data[high_vol_data['ma_buy']]
    ma_sell = high_vol_data[high_vol_data['ma_sell']]
    ax.plot(ma_buy.index, ma_buy['close'], 'o', markersize=8, color='cyan', label='Momentum Buy', markeredgecolor='k')
    ax.plot(ma_sell.index, ma_sell['close'], 'o', markersize=8, color='magenta', label='Momentum Sell', markeredgecolor='k')
    
    ax.set_title('Dynamic Backtest with GAM-Based Volume Regimes')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USDT)')
    ax.legend()
    ax.grid(True)
    plt.show()