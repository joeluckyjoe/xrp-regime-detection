import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
import yfinance as yf
import pandas_ta as ta
from collections import deque
import itertools
import matplotlib.pyplot as plt

# --- SKLEARN Imports ---
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

warnings.filterwarnings("ignore", category=FutureWarning)

# --- All Required Data and Feature Functions ---
def fetch_and_prepare_data(ticker="QQQ", period="729d", interval="1h"):
    base_df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
    if base_df.empty: return None
    vix_df = yf.Ticker("^VIX").history(period=period, interval="1d", auto_adjust=False)
    if vix_df.empty: base_df['vix'] = 20
    else:
        base_df.index = base_df.index.tz_convert('UTC'); vix_df.index = vix_df.index.tz_convert('UTC')
        vix_df.rename(columns={'Close': 'vix'}, inplace=True)
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
    return processed_dfs

class BayesianAnomalyDetector:
    def __init__(self, distribution_type='t'): self.dist_type = distribution_type; self.alpha_0, self.beta_0, self.mu_0, self.nu_0 = 1.0, 1.0, 0.0, 1.0; self.reset_posterior()
    def reset_posterior(self): self.alpha_n, self.beta_n, self.mu_n, self.nu_n = self.alpha_0, self.beta_0, self.mu_0, self.nu_0; self.mle_params = None
    def fit(self, data):
        if self.dist_type == 't': self._fit_t(data)
    def compute_surprise(self, x):
        from scipy.stats import t
        df,loc,scale=2*self.alpha_n,self.mu_n,np.sqrt(self.beta_n/(self.alpha_n*self.nu_n));
        if scale <= 0: return 0.5
        return 1.0 - (2 * min(t.cdf(x,df=df,loc=loc,scale=scale), t.sf(x,df=df,loc=loc,scale=scale)))
    def get_distribution_params(self): return 0, 1
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

def calculate_walk_forward_features(dataframes):
    timeframes=['1h','1d','1w']; indicators=['returns','Volume','rsi','atr','vix','MACD_12_26_9','MACDh_12_26_9','MACDs_12_26_9','ADX_14']
    dist_map={'returns':'t','Volume':'gamma','rsi':'beta','atr':'gamma','vix':'gamma','MACD_12_26_9':'t','MACDh_12_26_9':'t','MACDs_12_26_9':'t','ADX_14':'gamma'}
    window_sizes={'1h':882,'1d':126,'1w':26}; detectors,normalizers,data_windows={},{},{}
    for tf,ind in itertools.product(timeframes,indicators):
        key=f'{tf}_{ind}';win_size=window_sizes[tf]
        detectors[key]=BayesianAnomalyDetector(distribution_type=dist_map.get(ind, 't')); normalizers[f'{key}_raw']=RollingNormalizer(win_size)
        normalizers[f'{key}_mean']=RollingNormalizer(win_size); normalizers[f'{key}_scale']=RollingNormalizer(win_size); data_windows[key]=list(dataframes[tf][ind].iloc[:win_size])
    for key,window in data_windows.items():
        detectors[key].fit(pd.Series(window)); [normalizers[f'{key}_raw'].update(val) for val in window]
    all_features=[]; start_ts=dataframes['1w'].index[window_sizes['1w']]; start_index=dataframes['1h'].index.searchsorted(start_ts)
    for i in range(start_index,len(dataframes['1h'])):
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
        if len(step_features)==108: all_features.append(step_features)
    return np.array(all_features,dtype=np.float32), dataframes['1h'].iloc[start_index:start_index+len(all_features)]

# --- REGIME DETECTION EXPERIMENT FUNCTIONS ---

def generate_vix_regime_labels_quantile(df, bull_quantile=0.4, bear_quantile=0.8):
    print("Generating VIX Ground Truth Labels using Quantiles...")
    bull_threshold = df['vix'].quantile(bull_quantile)
    bear_threshold = df['vix'].quantile(bear_quantile)
    print(f"  VIX Thresholds: BULL < {bull_threshold:.2f}, BEAR > {bear_threshold:.2f} (based on full dataset)")
    labels = pd.Series('SIDEWAYS', index=df.index, dtype=str)
    labels[df['vix'] < bull_threshold] = 'BULL'
    labels[df['vix'] > bear_threshold] = 'BEAR'
    return labels

def run_static_supervised_model(features, df, curated_feature_indices, vix_labels, train_window_size):
    print("Running Baseline: Static Supervised Model...")
    curated_features = features[:, curated_feature_indices]
    
    # Train on the first window only
    X_train = curated_features[:train_window_size]
    y_train = vix_labels.iloc[:train_window_size]
    
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    
    model = LogisticRegression(random_state=42, class_weight='balanced').fit(X_train_scaled, y_train)
    
    # Predict on the rest of the data
    X_test = curated_features[train_window_size:]
    X_test_scaled = scaler.transform(X_test)
    
    predictions = model.predict(X_test_scaled)
    
    return pd.Series(predictions, index=df.index[train_window_size:])

def run_rolling_supervised_model(features, df, curated_feature_indices, train_window_size):
    print("Running Main Model: Rolling Supervised Model...")
    curated_features = features[:, curated_feature_indices]
    labels = pd.Series(index=df.index, dtype=str)

    for i in tqdm(range(train_window_size, len(df))):
        # Define the rolling window
        X_window = curated_features[i-train_window_size:i]
        df_window = df.iloc[i-train_window_size:i]
        
        # Generate VIX labels just for this window
        y_window = generate_vix_regime_labels_quantile(df_window)
        
        # Scale and train the model on this window
        scaler = StandardScaler().fit(X_window)
        X_window_scaled = scaler.transform(X_window)
        model = LogisticRegression(random_state=42, class_weight='balanced').fit(X_window_scaled, y_window)
        
        # Predict for the single current point
        current_point_scaled = scaler.transform(curated_features[i,:].reshape(1,-1))
        prediction = model.predict(current_point_scaled)[0]
        labels.iloc[i] = prediction
        
    return labels

def plot_regime_comparison(df, filename='regime_comparison_plot_v3.png'):
    print(f"Generating visual comparison plot: {filename}...")
    fig, axes = plt.subplots(3, 1, figsize=(20, 8), sharex=True)
    color_map = {'BULL': 'forestgreen', 'BEAR': 'crimson', 'SIDEWAYS': 'goldenrod'}
    
    models = ['VIX_GroundTruth', 'Static_Model', 'Rolling_Model']
    titles = ['VIX Ground Truth (Answer Key)', 'Baseline: Static Supervised Model', 'Main Model: Rolling Supervised Model']
    
    for ax, model, title in zip(axes, models, titles):
        # Fill NaNs for continuous plotting
        plot_series = df[model].ffill()
        for regime, color in color_map.items():
            for i in np.where(plot_series == regime)[0]:
                ax.axvspan(df.index[i], df.index[i] + pd.Timedelta(hours=1), color=color, alpha=0.6)
        ax.set_title(title)
        ax.set_yticks([])
        ax.margins(x=0)

    fig.tight_layout()
    plt.savefig(filename)
    plt.close()
    print("Plot saved.")

if __name__ == '__main__':
    TICKER = "QQQ"
    TRAIN_WINDOW_SIZE = 252 * 7 # Approx 1 year of hourly data
    
    print(f"--- Starting v3 Supervised Regime Detection for {TICKER} ---")
    data = fetch_and_prepare_data(ticker=TICKER)
    features, df = calculate_walk_forward_features(data)
    
    curated_indices = [
        (0*36) + (0*4) + 1, # returns_1h
        (1*36) + (5*4) + 1, # MACD_12_26_9_1d
        (0*36) + (4*4) + 1, # vix_1h
        (0*36) + (3*4) + 1, # atr_1h
        (1*36) + (8*4) + 1, # ADX_14_1d
        (1*36) + (2*4) + 1, # rsi_1d
    ]
    
    # --- Generate all three sets of labels ---
    vix_labels = generate_vix_regime_labels_quantile(df)
    static_labels = run_static_supervised_model(features, df, curated_indices, vix_labels, TRAIN_WINDOW_SIZE)
    rolling_labels = run_rolling_supervised_model(features, df, curated_indices, TRAIN_WINDOW_SIZE)
    
    # --- Create a comparison DataFrame ---
    results = pd.DataFrame({
        'VIX_GroundTruth': vix_labels,
        'Static_Model': static_labels,
        'Rolling_Model': rolling_labels
    }).dropna() # Drop NaNs from the start where models haven't predicted yet
    
    # --- Print Comparison Report ---
    print("\n\n--- REGIME DETECTION COMPARISON REPORT (v3) ---")
    print("Comparing model labels against VIX-based ground truth.\n")
    class_labels = ['BEAR', 'BULL', 'SIDEWAYS']
    
    # Static Model Report
    static_accuracy = accuracy_score(results['VIX_GroundTruth'], results['Static_Model'])
    static_cm = confusion_matrix(results['VIX_GroundTruth'], results['Static_Model'], labels=class_labels)
    print("--- Baseline: Static Supervised Model ---")
    print(f"Accuracy: {static_accuracy:.2%}")
    print("Confusion Matrix (Rows: VIX Truth, Cols: Model Prediction):")
    print("          BEAR   BULL   SIDEWAYS")
    print(f"BEAR    {static_cm[0,0]:<7d}{static_cm[0,1]:<7d}{static_cm[0,2]:<7d}")
    print(f"BULL    {static_cm[1,0]:<7d}{static_cm[1,1]:<7d}{static_cm[1,2]:<7d}")
    print(f"SIDEWAYS{static_cm[2,0]:<7d}{static_cm[2,1]:<7d}{static_cm[2,2]:<7d}")
    
    print("\n" + "-"*40 + "\n")
    
    # Rolling Model Report
    rolling_accuracy = accuracy_score(results['VIX_GroundTruth'], results['Rolling_Model'])
    rolling_cm = confusion_matrix(results['VIX_GroundTruth'], results['Rolling_Model'], labels=class_labels)
    print("--- Main Model: Rolling Supervised Model ---")
    print(f"Accuracy: {rolling_accuracy:.2%}")
    print("Confusion Matrix (Rows: VIX Truth, Cols: Model Prediction):")
    print("          BEAR   BULL   SIDEWAYS")
    print(f"BEAR    {rolling_cm[0,0]:<7d}{rolling_cm[0,1]:<7d}{rolling_cm[0,2]:<7d}")
    print(f"BULL    {rolling_cm[1,0]:<7d}{rolling_cm[1,1]:<7d}{rolling_cm[1,2]:<7d}")
    print(f"SIDEWAYS{rolling_cm[2,0]:<7d}{rolling_cm[2,1]:<7d}{rolling_cm[2,2]:<7d}")
    print("\n--- End of Report ---")

    # --- Generate Visual Plot ---
    plot_regime_comparison(results.reindex(df.index)) # Reindex to plot the full timeline