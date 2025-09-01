import pandas as pd
import pymc as pm
import numpy as np
import arviz as az
from sklearn.preprocessing import StandardScaler
import pytensor.tensor as pt
import warnings

warnings.filterwarnings("ignore")

def train_regime_model(features_path='data/informed_flow_features.csv', n_regimes=3):
    """
    Trains a Bayesian Gaussian Mixture Model to identify market regimes.
    """
    print(f"Loading features from '{features_path}'...")
    try:
        features_df = pd.read_csv(features_path, index_col=0, parse_dates=True)
    except FileNotFoundError:
        print(f"Error: The file '{features_path}' was not found.")
        print("Please run 'create_informed_flow_features.py' first.")
        return None

    # Scale the features for better model performance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_df)
    n_features = X_scaled.shape[1]

    print(f"Building Bayesian Gaussian Mixture Model with {n_regimes} regimes...")
    with pm.Model() as regime_model:
        # --- Priors ---
        # Prior for the weights of each regime.
        weights = pm.Dirichlet('weights', a=np.ones(n_regimes))

        # Priors for the means of each feature within each regime.
        means = pm.Normal('means', mu=0, sigma=1.5, shape=(n_regimes, n_features))

        # --- Covariance structure ---
        # 1. Define the Cholesky factor of the correlation matrix for each regime.
        chol_corr = pm.LKJCholeskyCov("chol_corr", n=n_features, eta=2.0, sd_dist=pm.Exponential.dist(1.0), shape=n_regimes)
        
        # 2. Define the standard deviations for each feature within each regime.
        sigmas = pm.Exponential("sigmas", 1.0, shape=(n_regimes, n_features))
        
        # --- FIX: Manually construct the full covariance matrix ---
        # This is a more robust method that avoids internal errors with the Cholesky factor.
        comp_dists = []
        for i in range(n_regimes):
            # Create a diagonal matrix from the standard deviations
            sigma_diag = pt.diag(sigmas[i])
            # Reconstruct the full correlation matrix from its Cholesky factor
            corr_matrix = pt.dot(chol_corr[i], chol_corr[i].T)
            # Construct the full covariance matrix
            cov = pt.dot(pt.dot(sigma_diag, corr_matrix), sigma_diag)
            
            # Now, create the distribution using mu and the full covariance matrix.
            dist = pm.MvNormal.dist(mu=means[i], cov=cov)
            comp_dists.append(dist)

        # --- Likelihood ---
        # The core of the model. We pass the list of component distributions.
        pm.Mixture(
            'likelihood',
            w=weights,
            comp_dists=comp_dists,
            observed=X_scaled
        )
        # --- END FIX ---

    print("Starting MCMC sampling... (This may take several minutes)")
    with regime_model:
        # Run the NUTS sampler to get the posterior distribution
        trace = pm.sample(2000, tune=1000, cores=1, return_inferencedata=True)
    
    print("Model training complete.")
    return trace

if __name__ == '__main__':
    # Set n_regimes to 3 for "Normal", "Informed Buying", "Informed Selling"
    trained_trace = train_regime_model(n_regimes=3)
    
    if trained_trace is not None:
        # Save the trained model so we can use it later without retraining
        output_path = 'data/regime_model_trace.nc'
        trained_trace.to_netcdf(output_path)
        print(f"\nSuccessfully saved trained model trace to '{output_path}'")
        
        # Optional: Print a summary of the model results
        print("\nModel Summary:")
        print(az.summary(trained_trace, var_names=['weights', 'means']))

