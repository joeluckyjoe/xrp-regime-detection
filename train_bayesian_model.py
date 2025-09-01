import pandas as pd
import pymc as pm
import numpy as np
import arviz as az
import warnings
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    print("Loading training data...")
    features_df = pd.read_csv('data/nasdaq_features.csv', index_col=0, parse_dates=True)
    target_df = pd.read_csv('data/nasdaq_target.csv', index_col=0, parse_dates=True)

    # Scale the features for better model performance
    scaler = StandardScaler()
    X = scaler.fit_transform(features_df)
    y = target_df['target'].values
    
    print("Building Bayesian Logistic Regression model...")
    with pm.Model() as logistic_model:
        # Priors for the model coefficients (one for each feature)
        betas = pm.Normal('betas', mu=0, sigma=1, shape=X.shape[1])
        # Prior for the intercept
        intercept = pm.Normal('intercept', mu=0, sigma=1)
        
        # Linear combination of features and coefficients
        logits = pm.math.dot(X, betas) + intercept
        
        # Likelihood (Bernoulli for binary 0/1 outcome)
        likelihood = pm.Bernoulli('likelihood', logit_p=logits, observed=y)
        
        print("Sampling from the model (this will take a few minutes)...")
        trace = pm.sample(2000, tune=1500, chains=4, cores=1)
    
    print("\nModel training complete!")
    print("\nThis summary shows the learned importance of each feature.")
    print("If a feature's distribution is far from zero, it's an important predictor.")
    print(az.summary(trace, var_names=['intercept', 'betas']))
    
    # Save the trained model for use in our live bot
    trace.to_netcdf("data/bayesian_signal_model.nc")
    print("\nModel saved to 'data/bayesian_signal_model.nc'")