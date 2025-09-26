import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
import itertools
import os
import pickle
import yfinance as yf
from collections import deque

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Configuration ---
TICKERS = ["QQQ"]
OUTPUT_DIR = "final_labeled_data"
HMM_MODEL_PATH = "hmmlearn_final_results/hmm_model.pkl"
VOLATILITY_WINDOW = 24

# --- HELPER FUNCTIONS TO REPLACE PANDAS-TA ---
# (Helper functions are unchanged)
def calculate_rsi(close, length=14):
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))
def calculate_atr(high, low, close, length=14):
    tr1 = pd.DataFrame(high - low); tr2 = pd.DataFrame(abs(high - close.shift(1))); tr3 = pd.DataFrame(abs(low - close.shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/length, adjust=False).mean()
    return atr / close * 100
def calculate_macd(close, fast=12, slow=26, signal=9):
    exp1 = close.ewm(span=fast, adjust=False).mean(); exp2 = close.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2; signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram
def calculate_adx(high, low, close, length=14):
    plus_dm = high.diff(); minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0; minus_dm[minus_dm > 0] = 0
    tr1 = pd.DataFrame(high - low); tr2 = pd.DataFrame(abs(high - close.shift(1))); tr3 = pd.DataFrame(abs(low - close.shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/length, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/length, adjust=False).mean() / atr)
    minus_di = 100 * (abs(minus_dm.ewm(alpha=1/length, adjust=False).mean()) / atr)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = dx.ewm(alpha=1/length, adjust=False).mean()
    return adx

# --- LAYER 1: FEATURE ENGINEERING CODE ---
# (Classes and functions are unchanged)
class BayesianAnomalyDetector:
    def __init__(self, distribution_type='t'): self.dist_type = distribution_type; self.alpha_0, self.beta_0, self.mu_0, self.nu_0 = 1.0, 1.0, 0.0, 1.0; self.reset_posterior()
    def reset_posterior(self): self.alpha_n, self.beta_n, self.mu_n, self.nu_n = self.alpha_0, self.beta_0, self.mu_0, self.nu_0
    def fit(self, data):
        if self.dist_type == 't': self._fit_t(data)
    def compute_surprise(self, x):
        from scipy.stats import t
        df,loc,scale=2*self.alpha_n,self.mu_n,np.sqrt(self.beta_n/(self.alpha_n*self.nu_n));
        if scale <= 0 or not np.isfinite(scale): return 0.5
        return 1.0 - (2 * min(t.cdf(x,df=df,loc=loc,scale=scale), t.sf(x,df=df,loc=loc,scale=scale)))
    def get_distribution_params(self):
        scale = np.sqrt(self.beta_n / (self.alpha_n * self.nu_n)); return self.mu_n, scale
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
def fetch_and_prepare_data(tickers=["QQQ"], period="729d", interval="1h"):
    print(f"--- Fetching and Preparing Multi-Scale Data for {', '.join(tickers)} ---")
    all_asset_data = {}
    for ticker in tickers:
        base_df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
        if base_df.empty: continue
        agg_logic = {'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}
        df_1d = base_df.resample('D').agg(agg_logic).dropna(); df_1w = base_df.resample('W-MON').agg(agg_logic).dropna()
        dataframes = {'1h': base_df, '1d': df_1d, '1w': df_1w}; processed_dfs = {}
        for timeframe, df in dataframes.items():
            df_processed = df.copy(); df_processed['returns'] = (df_processed['Close'] - df_processed['Open']) / df_processed['Open']
            df_processed['rsi'] = calculate_rsi(df_processed['Close']) / 100.0
            df_processed['atr'] = calculate_atr(df_processed['High'], df_processed['Low'], df_processed['Close'])
            macd, macds, macdh = calculate_macd(df_processed['Close'])
            df_processed['MACD_12_26_9'] = macd; df_processed['MACDs_12_26_9'] = macds; df_processed['MACDh_12_26_9'] = macdh
            df_processed['ADX_14'] = calculate_adx(df_processed['High'], df_processed['Low'], df_processed['Close'])
            df_processed.dropna(inplace=True)
            required_cols = ['Open','High','Low','Close','Volume','returns','rsi','atr','MACD_12_26_9','MACDh_12_26_9','MACDs_12_26_9','ADX_14']
            processed_dfs[timeframe] = df_processed[required_cols]
        all_asset_data[ticker] = processed_dfs
    return all_asset_data
def calculate_walk_forward_features(all_asset_data):
    print("--- Calculating Layer 1 Walk-Forward Features ---")
    all_features_dict = {}; aligned_dfs_dict = {}
    for ticker, dataframes in all_asset_data.items():
        timeframes=['1h','1d','1w']; indicators=['returns','Volume','rsi','atr', 'MACD_12_26_9','MACDh_12_26_9','MACDs_12_26_9','ADX_14']
        window_sizes={'1h':882,'1d':126,'1w':26}; detectors,normalizers,data_windows={},{},{}
        for tf,ind in itertools.product(timeframes,indicators):
            key=f'{tf}_{ind}';win_size=window_sizes[tf]
            detectors[key]=BayesianAnomalyDetector(distribution_type='t'); normalizers[f'{key}_raw']=RollingNormalizer(win_size)
            normalizers[f'{key}_mean']=RollingNormalizer(win_size); normalizers[f'{key}_scale']=RollingNormalizer(win_size); 
            data_windows[key]=list(dataframes[tf][ind].iloc[:win_size])
        for key,window in data_windows.items():
            detectors[key].fit(pd.Series(window)); [normalizers[f'{key}_raw'].update(val) for val in window]
        start_ts=dataframes['1w'].index[window_sizes['1w']]; start_index=dataframes['1h'].index.searchsorted(start_ts)
        asset_features=[]
        for i in tqdm(range(start_index,len(dataframes['1h'])), desc=f"  Calculating {ticker} Features"):
            step_features=[]; current_ts=dataframes['1h'].index[i]
            for tf in timeframes:
                tf_loc=dataframes[tf].index.searchsorted(current_ts,side='right')-1
                if tf_loc < 0: continue
                current_observation=dataframes[tf].iloc[tf_loc]
                for ind in indicators:
                    key=f'{tf}_{ind}';current_value=current_observation[ind]
                    if len(data_windows[key]) <= tf_loc:
                        data_windows.setdefault(key, []).append(current_value)
                        if len(data_windows[key])>window_sizes[tf]: data_windows[key].pop(0)
                        detectors[key].fit(pd.Series(data_windows[key]))
                    surprise=detectors[key].compute_surprise(current_value); mean,scale=detectors[key].get_distribution_params()
                    normalizers[f'{key}_raw'].update(current_value); norm_raw=normalizers[f'{key}_raw'].normalize(current_value)
                    normalizers[f'{key}_mean'].update(mean); norm_mean=normalizers[f'{key}_mean'].normalize(mean)
                    normalizers[f'{key}_scale'].update(scale); norm_scale=normalizers[f'{key}_scale'].normalize(scale)
                    step_features.extend([surprise,norm_raw,norm_mean,norm_scale])
            if len(step_features) == 96: asset_features.append(step_features)
        feature_array = np.array(asset_features, dtype=np.float32)
        all_features_dict[ticker] = feature_array
        aligned_dfs_dict[ticker] = dataframes['1h'].iloc[start_index:start_index+len(asset_features)]
    return all_features_dict, aligned_dfs_dict

# --- Main Integration Execution ---
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Run Layer 1
    raw_data = fetch_and_prepare_data(tickers=TICKERS)
    features, aligned_dfs = calculate_walk_forward_features(raw_data)
    
    # 2. Run Layer 2
    print("\n--- Loading HMM Model and Labeling Regimes ---")
    with open(HMM_MODEL_PATH, "rb") as f:
        hmm_model = pickle.load(f)

    for ticker in TICKERS:
        print(f"Processing ticker: {ticker}")
        # This is the original dataframe aligned with the features
        df = aligned_dfs[ticker]
        
        # --- FIX: Perform a robust alignment of volatility and regimes ---
        # Calculate volatility but DON'T drop NaNs yet
        returns = df['Close'].pct_change()
        volatility = returns.rolling(window=VOLATILITY_WINDOW).std()
        
        # Create a new dataframe with just the data we need for labeling
        labeling_df = pd.DataFrame({'volatility': volatility})
        
        # Drop rows where volatility could not be calculated
        labeling_df.dropna(inplace=True)
        
        # Predict regimes on the valid volatility data
        regimes_raw = hmm_model.predict(labeling_df['volatility'].values.reshape(-1, 1))
        
        # Enforce consistent labeling
        low_vol_state = np.argmin(hmm_model.means_)
        regimes_ordered = np.where(regimes_raw == low_vol_state, 0, 1) # 0=COMPRESSION, 1=EXPANSiON
        
        # Add the final labels to our labeling_df
        labeling_df['regime'] = regimes_ordered
        
        # Join the final labels back to the main dataframe
        # This correctly aligns everything, leaving NaNs at the start of the 'regime' column
        df_labeled = df.join(labeling_df['regime'])
        
        # Now, we trim the start of all dataframes and arrays to match the first valid regime
        first_valid_regime_idx = df_labeled['regime'].first_valid_index()
        df_labeled = df_labeled.loc[first_valid_regime_idx:].copy()
        df_labeled['regime'] = df_labeled['regime'].astype(int)
        
        # Align the feature array with the final labeled dataframe
        final_features = features[ticker][-len(df_labeled):]
        final_labels = df_labeled['regime'].values
        
        # Final sanity check
        if len(final_features) != len(final_labels):
            raise ValueError("Final alignment between features and labels failed. This should not happen.")

        # 3. Save Final Datasets
        print(f"Saving final labeled data for {ticker}...")
        
        csv_path = os.path.join(OUTPUT_DIR, f"{ticker}_labeled_market_data.csv")
        df_labeled.to_csv(csv_path)
        print(f"  Saved market data with labels to {csv_path}")
        
        npz_path = os.path.join(OUTPUT_DIR, f"{ticker}_features_and_labels.npz")
        np.savez_compressed(npz_path, features=final_features, labels=final_labels)
        print(f"  Saved features and labels to {npz_path}")

    print("\n✅ Integration and labeling complete.")