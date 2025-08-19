# XRP/USDT Market Regime Detection using Bayesian Change Point Analysis

This project provides a visual tool to identify market regime changes in the XRP/USDT trading pair using Bayesian Change Point Detection with PyMC.

## Project Structure

```
xrp-regime-detection/
│
├── data/
│   └── xrp_usdt_data.csv
│
├── notebooks/
│   └── xrp_regime_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_fetcher.py
│   └── visualization.py
│
└── README.md
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd xrp-regime-detection
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS and Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You will need to create a `requirements.txt` file using `pip freeze > requirements.txt`)*

## How to Run

1.  **Fetch the latest data:**
    ```bash
    python src/data_fetcher.py
    ```

2.  **Open and run the Jupyter Notebook:**
    ```bash
    jupyter lab notebooks/xrp_regime_analysis.ipynb
    ```

    Run the cells in the notebook to perform the analysis and generate the visualizations.

## Interpreting the Results

* **Price and Log Returns Plot:** Provides a basic understanding of the price action and volatility.
* **Posterior Probability of Change Points:** The main output of the Bayesian model. Peaks in this plot indicate a high probability of a regime change at that point in time.
* **XRP/USDT Price with Detected Regimes:** This plot overlays the detected regimes on the price chart. The color of the shaded areas indicates the nature of the regime (e.g., green for positive average returns, red for negative). This provides a clear visual signal of when the market dynamics have shifted.