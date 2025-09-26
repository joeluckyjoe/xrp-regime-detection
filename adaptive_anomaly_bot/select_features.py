import pandas as pd
import numpy as np
import warnings
import itertools
from tqdm import tqdm
from collections import deque

import yfinance as yf
import pandas_ta as ta
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler

# --- Suppress Warnings ---
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
ASSET_TO_ANALYZE = "USO"
TOP_N_FEATURES = 18

# --- DATA AND FEATURE GENERATION CODE (Copied from main script) ---
# NOTE: This section is a direct copy of the functions from our main script
# to ensure the data is generated in the exact same way.

def fetch_and_prepare_data(tickers=["QQQ"], period="729d", interval="1h"):
    print(f"--- Fetching and Preparing Multi-Scale Data for {', '.join(tickers)} ---")
    vix_df = yf.Ticker("^VIX").history(period=period, interval="1d", auto_adjust=False)
    if not vix_df.empty:
        vix_df.index = vix_df.index.tz_convert('UTC')
        vix_df.rename(columns={'Close': 'vix'}, inplace=True)
    all_asset_data = {}
    for ticker in tickers:
        base_df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
        if base_df.empty: continue
        if vix_df.empty: base_df['vix'] = 20
        else:
            base_df.index = base_df.index.tz_convert('UTC')
            base_df = pd.merge_asof(left=base_df.sort_index(), right=vix_df[['vix']].sort_index(), left_index=True, right_index=True, direction='backward')
        agg_logic = {'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum','vix':'last'}
        df_1d = base_df.resample('D').agg(agg_logic).dropna(); df_1w = base_df.resample('W-MON').agg(agg_logic).dropna()
        dataframes = {'1h': base_df, '1d': df_1d, '1w': df_1w}; processed_dfs = {}
        for timeframe, df in dataframes.items():
            df_processed = df.copy(); df_processed['returns'] = (df_processed['Close']-df_processed['Open'])/df_processed['Open']
            df_processed.ta.rsi(length=14, append=True); df_processed.ta.atr(length=14, append=True); df_processed.ta.macd(append=True); df_processed.ta.adx(append=True)
            df_processed.rename(columns={"RSI_14":"rsi", "ATRr_14":"atr"}, inplace=True); df_processed['rsi'] = df_processed['rsi'] / 100.0; df_processed.dropna(inplace=True)
            required_cols = ['Open','High','Low','Close','Volume','returns','rsi','atr','vix', 'MACD_12_26_9','MACDh_12_26_9','MACDs_12_26_9','ADX_14']
            processed_dfs[timeframe] = df_processed[required_cols]
        all_asset_data[ticker] = processed_dfs
    return all_asset_data

class BayesianAnomalyDetector:
    def __init__(self, distribution_type='t'): self.dist_type = distribution_type; self.alpha_0, self.beta_0, self.mu_0, self.nu_0 = 1.0, 1.0, 0.0, 1.0; self.reset_posterior()
    def reset_posterior(self): self.alpha_n, self.beta_n, self.mu_n, self.nu_n = self.alpha_0, self.beta_0, self.mu_0, self.nu_0
    def fit(self, data):
        if self.dist_type == 't': self._fit_t(data)
    def compute_surprise(self, x):
        from scipy.stats import t
        df,loc,scale=2*self.alpha_n,self.mu_n,np.sqrt(self.beta_n/(self.alpha_n*self.nu_n));
        if scale <= 0: return 0.5
        return 1.0 - (2 * min(t.cdf(x,df=df,loc=loc,scale=scale), t.sf(x,df=df,loc=loc,scale=scale)))
    def get_distribution_params(self): return self.mu_n, np.sqrt(self.beta_n/(self.alpha_n*self.nu_n))
    def _fit_t(self, data):
        n=len(data);
        if n==0: return
        mean_data,sum_sq_diff=data.mean(),((data-data.mean())**2).sum()
        self.alpha_n=self.alpha_0+n/2; self.beta_n=self.beta_0+0.5*sum_sq_diff+(n*self.nu_0)/(self.nu_0+n)*0.5*(mean_data-self.mu_0)**2
        self.mu_n=(self.nu_0*self.mu_0+n*mean_data)/(self.nu_0+n); self.nu_n=self.nu_0+n

class RollingNormalizer:
    def __init__(self, window_size=252): self.window_size, self.data, self.min, self.max = window_size, deque(maxlen=window_size), None, None
    def update(self, value): self.data.append(value); self.min, self.max = min(self.data), max(self.data)
    def normalize(self, value): return 0.5 if self.max is None or self.min is None or self.max == self.min else (value - self.min) / (self.max - self.min)

def calculate_walk_forward_features(all_asset_data):
    all_features_dict = {}
    aligned_dfs_dict = {}
    for ticker, dataframes in all_asset_data.items():
        timeframes=['1h','1d','1w']; indicators=['returns','Volume','rsi','atr','vix', 'MACD_12_26_9','MACDh_12_26_9','MACDs_12_26_9','ADX_14']
        window_sizes={'1h':882,'1d':126,'1w':26}; detectors,normalizers,data_windows={},{},{}
        for tf,ind in itertools.product(timeframes,indicators):
            key=f'{tf}_{ind}';win_size=window_sizes[tf]
            detectors[key]=BayesianAnomalyDetector(distribution_type='t'); normalizers[f'{key}_raw']=RollingNormalizer(win_size)
            normalizers[f'{key}_mean']=RollingNormalizer(win_size); normalizers[f'{key}_scale']=RollingNormalizer(win_size); data_windows[key]=list(dataframes[tf][ind].iloc[:win_size])
        for key,window in data_windows.items():
            detectors[key].fit(pd.Series(window)); [normalizers[f'{key}_raw'].update(val) for val in window]
        asset_features=[]; start_ts=dataframes['1w'].index[window_sizes['1w']]; start_index=dataframes['1h'].index.searchsorted(start_ts)
        for i in tqdm(range(start_index,len(dataframes['1h'])), desc=f"  Calculating {ticker} Features"):
            step_features=[]; current_ts=dataframes['1h'].index[i]
            for tf in timeframes:
                tf_loc=dataframes[tf].index.searchsorted(current_ts,side='right')-1; current_observation=dataframes[tf].iloc[tf_loc]
                for ind in indicators:
                    key=f'{tf}_{ind}';current_value=current_observation[ind]
                    if current_ts>=dataframes[tf].index[tf_loc] and len(data_windows[key])<=tf_loc:
                        data_windows[key].append(current_value)
                        if len(data_windows[key])>window_sizes[tf]: data_windows[key].pop(0)
                        detectors[key].fit(pd.Series(data_windows[key]))
                    surprise=detectors[key].compute_surprise(current_value); mean,scale=detectors[key].get_distribution_params()
                    normalizers[f'{key}_raw'].update(current_value); norm_raw=normalizers[f'{key}_raw'].normalize(current_value)
                    normalizers[f'{key}_mean'].update(mean); norm_mean=normalizers[f'{key}_mean'].normalize(mean)
                    normalizers[f'{key}_scale'].update(scale); norm_scale=normalizers[f'{key}_scale'].normalize(scale)
                    step_features.extend([surprise,norm_raw,norm_mean,norm_scale])
            if len(step_features)==108: asset_features.append(step_features)
        all_features_dict[ticker] = np.array(asset_features,dtype=np.float32)
        aligned_dfs_dict[ticker] = dataframes['1h'].iloc[start_index:start_index+len(asset_features)]
    return all_features_dict, aligned_dfs_dict

def generate_atr_regime_labels_quantile(df, bull_quantile=0.4, bear_quantile=0.8):
    bull_threshold = df['atr'].quantile(bull_quantile)
    bear_threshold = df['atr'].quantile(bear_quantile)
    labels = pd.Series('SIDEWAYS', index=df.index, dtype=str)
    labels[df['atr'] < bull_threshold] = 'BULL'
    labels[df['atr'] > bear_threshold] = 'BEAR'
    return labels

def main():
    """
    Main function to perform feature selection for a specific asset.
    """
    print(f"--- Starting Data-Driven Feature Selection for {ASSET_TO_ANALYZE} ---")

    # 1. Generate the full dataset
    all_data = fetch_and_prepare_data(tickers=[ASSET_TO_ANALYZE])
    if not all_data:
        print(f"Could not fetch data for {ASSET_TO_ANALYZE}. Exiting.")
        return

    features_dict, dfs_dict = calculate_walk_forward_features(all_data)
    full_df = dfs_dict[ASSET_TO_ANALYZE]
    all_features = features_dict[ASSET_TO_ANALYZE]

    # 2. Assemble the complete training dataset from all windows
    train_ws, test_ws = 252 * 7, 21 * 7
    num_w = (len(all_features) - train_ws) // test_ws

    all_training_features = []
    all_training_labels = []

    print(f"\n--- Assembling training data from {num_w} walk-forward windows... ---")
    for i in range(num_w):
        train_si, train_ei = i * test_ws, i * test_ws + train_ws
        
        train_df_window = full_df.iloc[train_si:train_ei]
        train_features_window = all_features[train_si:train_ei]
        
        y_window = generate_atr_regime_labels_quantile(train_df_window)
        
        all_training_features.append(train_features_window)
        all_training_labels.append(y_window)

    X_full = np.concatenate(all_training_features)
    y_full = pd.concat(all_training_labels).values

    # Scale features for stability
    scaler = StandardScaler()
    X_full_scaled = scaler.fit_transform(X_full)

    # 3. Perform the feature selection
    print(f"\n--- Running Feature Selection to find the best {TOP_N_FEATURES} features... ---")
    selector = SelectKBest(mutual_info_classif, k=TOP_N_FEATURES)
    selector.fit(X_full_scaled, y_full)
    
    # 4. Get and display the results
    top_indices = selector.get_support(indices=True)
    scores = selector.scores_[top_indices]
    
    # Create a sorted list of (score, index)
    sorted_features = sorted(zip(scores, top_indices), reverse=True)

    # Define feature names for easy interpretation
    timeframes = ['1h', '1d', '1w']
    indicators = ['returns','Volume','rsi','atr','vix', 'MACD','MACDh','MACDs','ADX']
    feature_types = ['surprise', 'norm_raw', 'norm_mean', 'norm_scale']
    
    feature_names = []
    for tf in timeframes:
        for ind in indicators:
            for f_type in feature_types:
                feature_names.append(f"{tf}_{ind}_{f_type}")

    print(f"\n--- Top {TOP_N_FEATURES} Most Predictive Features for {ASSET_TO_ANALYZE} ---")
    print("-" * 60)
    print(f"{'Rank':<5} {'Index':<7} {'Score':<10} {'Feature Name'}")
    print("-" * 60)
    for rank, (score, index) in enumerate(sorted_features, 1):
        print(f"{rank:<5} {index:<7} {score:<10.4f} {feature_names[index]}")
    print("-" * 60)

    print("\n--- Python list of best indices ---")
    best_indices_list = [index for _, index in sorted_features]
    print(best_indices_list)
    print("\nFeature selection complete. You can now use this list of indices in the main script.")

if __name__ == '__main__':
    main()