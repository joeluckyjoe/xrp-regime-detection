import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from scipy.stats import t

class BayesianAnomalyDetector:
    """
    Models the distribution of a data series using a Bayesian approach
    with a Normal-Gamma prior, resulting in a Student's t-distribution
    for the posterior predictive.
    """
    def __init__(self, alpha_0=1.0, beta_0=1.0, mu_0=0.0, nu_0=1.0):
        # Hyperparameters for the Normal-Gamma prior
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        self.mu_0 = mu_0
        self.nu_0 = nu_0
        self.reset_posterior()

    def reset_posterior(self):
        """Resets the posterior parameters to the prior."""
        self.alpha_n = self.alpha_0
        self.beta_n = self.beta_0
        self.mu_n = self.mu_0
        self.nu_n = self.nu_0

    def fit(self, data):
        """
        Updates the posterior parameters based on the provided data.

        Args:
            data (pd.Series): The data series to fit the model on.
        """
        n = len(data)
        if n == 0:
            return

        mean_data = data.mean()
        sum_sq_diff = ((data - mean_data) ** 2).sum()

        # Update the posterior parameters based on standard Bayesian formulas
        self.alpha_n = self.alpha_0 + n / 2
        self.beta_n = self.beta_0 + 0.5 * sum_sq_diff + (n * self.nu_0) / (self.nu_0 + n) * 0.5 * (mean_data - self.mu_0)**2
        self.mu_n = (self.nu_0 * self.mu_0 + n * mean_data) / (self.nu_0 + n)
        self.nu_n = self.nu_0 + n

    def compute_surprise(self, x):
        """
        Computes the 'surprise score' for a new data point x.
        A score near 1.0 is a surprise; a score near 0.0 is normal.
        """
        df = 2 * self.alpha_n
        loc = self.mu_n
        scale = np.sqrt(self.beta_n / (self.alpha_n * self.nu_n))

        cdf_val = t.cdf(x, df=df, loc=loc, scale=scale)
        sf_val = t.sf(x, df=df, loc=loc, scale=scale)
        
        return 1.0 - (2 * min(cdf_val, sf_val))

def fetch_and_prepare_data(ticker="QQQ", period="2y", interval="1h"):
    """
    Fetches historical market data and calculates technical indicators.
    """
    print(f"Fetching {interval} data for {ticker} over the last {period}...")
    data = yf.Ticker(ticker).history(period=period, interval=interval)
    if data.empty:
        print(f"No data found for {ticker}.")
        return None
    
    print("Data fetched. Calculating indicators...")

    # ==> NEW: Calculate hourly returns <==
    data['returns'] = (data['Close'] - data['Open']) / data['Open']
    
    # Calculate other indicators
    data.ta.rsi(length=14, append=True)
    data.ta.atr(length=14, append=True)
    
    data.dropna(inplace=True)
    data.rename(columns={"RSI_14": "rsi", "ATRr_14": "atr"}, inplace=True)
    
    # ==> UPDATED: Include 'returns' in the final feature set <==
    features = data[['Open', 'High', 'Low', 'Close', 'Volume', 'returns', 'rsi', 'atr']]
    
    print("Indicators calculated.")
    return features

if __name__ == "__main__":
    feature_df = fetch_and_prepare_data()

    if feature_df is not None:
        print("\n--- DEMONSTRATION OF LAYER 1: SURPRISE SCORE CALCULATION ---")
        
        training_window = feature_df.iloc[-442:-1]
        last_observation = feature_df.iloc[-1]
        print(f"\nTraining on {len(training_window)} data points.")
        print("Scoring the last observation from:", last_observation.name)

        detectors = {}
        # ==> UPDATED: Add 'returns' to the list of features to model <==
        features_to_model = ['Volume', 'returns', 'rsi', 'atr']
        
        for feature in features_to_model:
            detectors[feature] = BayesianAnomalyDetector()
            detectors[feature].fit(training_window[feature])

        surprise_scores = {}
        for feature, detector in detectors.items():
            value = last_observation[feature]
            surprise_scores[f"{feature}_surprise"] = detector.compute_surprise(value)

        print("\n--- State Vector (Surprise Scores) ---")
        print(pd.Series(surprise_scores).round(4))