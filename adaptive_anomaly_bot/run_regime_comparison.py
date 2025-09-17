import pandas as pd
import numpy as np
import warnings
import random
import torch
from tqdm import tqdm
import yfinance as yf
import pandas_ta as ta
from collections import deque
import itertools

# --- SKLEARN Imports ---
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score

warnings.filterwarnings("ignore", category=RuntimeWarning, module='scipy.stats._continuous_distns')
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn.cluster._kmeans')
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
def generate_vix_regime_labels(df, bull_threshold, bear_threshold):
    print("Generating VIX Ground Truth Labels...")
    labels = pd.Series(index=df.index, dtype=str)
    labels[df['vix'] < bull_threshold] = 'BULL'
    labels[df['vix'] > bear_threshold] = 'BEAR'
    labels[(df['vix'] >= bull_threshold) & (df['vix'] <= bear_threshold)] = 'SIDEWAYS'
    return labels

def run_kmeans_model(df):
    print("Running Method 1: Rolling K-Means...")
    def create_fingerprint(window_df):
        vol = window_df['Close'].pct_change().std()
        log_price = np.log(window_df['Close'].replace(0, 1e-9))
        slope, _ = np.linalg.lstsq(np.vstack([np.arange(len(log_price)), np.ones(len(log_price))]).T, log_price, rcond=None)[0]
        return np.array([vol, slope])

    labels = pd.Series(index=df.index, dtype=str)
    window_size = 252 * 7 
    for i in tqdm(range(window_size, len(df))):
        window = df.iloc[i-window_size:i]
        fingerprints = np.array([create_fingerprint(window.iloc[j-100:j]) for j in range(100, len(window), 21)])
        scaler = StandardScaler().fit(fingerprints)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(scaler.transform(fingerprints))
        
        centers = scaler.inverse_transform(kmeans.cluster_centers_)
        bull_idx = np.argmax(centers[:, 1])
        bear_idx = np.argmin(centers[:, 1])
        
        current_fingerprint = create_fingerprint(window).reshape(1, -1)
        pred_label = kmeans.predict(scaler.transform(current_fingerprint))[0]
        
        if pred_label == bull_idx: labels.iloc[i] = 'BULL'
        elif pred_label == bear_idx: labels.iloc[i] = 'BEAR'
        else: labels.iloc[i] = 'SIDEWAYS'
    return labels.bfill()

def run_adaptive_gmm_model(features, df):
    print("Running Method 2: Adaptive GMM + PCA...")
    labels = pd.Series(index=df.index, dtype=str)
    window_size = 252 * 7 
    
    pca = PCA(n_components=10)
    initial_features = pca.fit_transform(features[:window_size])
    gmm = GaussianMixture(n_components=3, random_state=42, n_init=10).fit(initial_features)
    
    initial_labels = gmm.predict(initial_features)
    df_initial = df.iloc[:window_size]
    bull_idx = pd.Series(df_initial['Close'].pct_change().groupby(initial_labels).mean()).idxmax()
    bear_idx = pd.Series(df_initial['Close'].pct_change().groupby(initial_labels).mean()).idxmin()

    prev_weights, prev_means, prev_covs = gmm.weights_, gmm.means_, gmm.covariances_
    
    for i in tqdm(range(window_size, len(features))):
        current_point_pca = pca.transform(features[i,:].reshape(1, -1))
        window_features = pca.transform(features[i-window_size:i])
        
        # +++ FIX: Add regularization to prevent matrix inversion errors +++
        reg_cov = 1e-6 
        prev_covs_reg = prev_covs + np.eye(prev_covs.shape[1]) * reg_cov
        
        gmm = GaussianMixture(n_components=3, random_state=42,
                              weights_init=prev_weights,
                              means_init=prev_means,
                              precisions_init=np.linalg.inv(prev_covs_reg)).fit(window_features)
        
        pred_label = gmm.predict(current_point_pca)[0]
        
        if pred_label == bull_idx: labels.iloc[i] = 'BULL'
        elif pred_label == bear_idx: labels.iloc[i] = 'BEAR'
        else: labels.iloc[i] = 'SIDEWAYS'
            
        prev_weights, prev_means, prev_covs = gmm.weights_, gmm.means_, gmm.covariances_
        
    return labels.bfill()

if __name__ == '__main__':
    TICKER = "QQQ"
    VIX_BULL_THRESHOLD = 18
    VIX_BEAR_THRESHOLD = 30
    
    print(f"--- Starting Regime Detection Comparison for {TICKER} ---")
    data = fetch_and_prepare_data(ticker=TICKER)
    features, df = calculate_walk_forward_features(data)
    
    common_index = df.index
    
    vix_labels = generate_vix_regime_labels(df, VIX_BULL_THRESHOLD, VIX_BEAR_THRESHOLD)
    kmeans_labels = run_kmeans_model(df)
    gmm_labels = run_adaptive_gmm_model(features, df)
    
    results = pd.DataFrame({
        'VIX_GroundTruth': vix_labels,
        'KMeans_Labels': kmeans_labels,
        'GMM_Labels': gmm_labels
    }).dropna()
    
    print("\n\n--- REGIME DETECTION COMPARISON REPORT ---")
    print("Comparing model labels against VIX-based ground truth.\n")
    
    class_labels = ['BEAR', 'BULL', 'SIDEWAYS']
    
    # K-Means Report
    kmeans_accuracy = accuracy_score(results['VIX_GroundTruth'], results['KMeans_Labels'])
    kmeans_cm = confusion_matrix(results['VIX_GroundTruth'], results['KMeans_Labels'], labels=class_labels)
    print("--- Method 1: Rolling K-Means ---")
    print(f"Accuracy: {kmeans_accuracy:.2%}")
    print("Confusion Matrix (Rows: VIX Truth, Cols: Model Prediction):")
    print("          BEAR   BULL   SIDEWAYS")
    print(f"BEAR    {kmeans_cm[0,0]:<7d}{kmeans_cm[0,1]:<7d}{kmeans_cm[0,2]:<7d}")
    print(f"BULL    {kmeans_cm[1,0]:<7d}{kmeans_cm[1,1]:<7d}{kmeans_cm[1,2]:<7d}")
    print(f"SIDEWAYS{kmeans_cm[2,0]:<7d}{kmeans_cm[2,1]:<7d}{kmeans_cm[2,2]:<7d}")
    
    print("\n" + "-"*40 + "\n")
    
    # GMM Report
    gmm_accuracy = accuracy_score(results['VIX_GroundTruth'], results['GMM_Labels'])
    gmm_cm = confusion_matrix(results['VIX_GroundTruth'], results['GMM_Labels'], labels=class_labels)
    print("--- Method 2: Adaptive GMM + PCA ---")
    print(f"Accuracy: {gmm_accuracy:.2%}")
    print("Confusion Matrix (Rows: VIX Truth, Cols: Model Prediction):")
    print("          BEAR   BULL   SIDEWAYS")
    print(f"BEAR    {gmm_cm[0,0]:<7d}{gmm_cm[0,1]:<7d}{gmm_cm[0,2]:<7d}")
    print(f"BULL    {gmm_cm[1,0]:<7d}{gmm_cm[1,1]:<7d}{gmm_cm[1,2]:<7d}")
    print(f"SIDEWAYS{gmm_cm[2,0]:<7d}{gmm_cm[2,1]:<7d}{gmm_cm[2,2]:<7d}")
    print("\n--- End of Report ---")