import os
import pandas as pd
from polygon import RESTClient
from datetime import date, timedelta

def check_polygon_connection():
    """Connects to Polygon.io and fetches one day of NASDAQ data."""
    try:
        api_key = os.environ.get("POLYGON_API_KEY")
        if not api_key:
            print("\n❌ ERROR: POLYGON_API_KEY environment variable not set.")
            print("Please run 'export POLYGON_API_KEY=...' in your terminal first.")
            return

        print("Connecting to Polygon.io...")
        
        # --- THIS IS THE FIX ---
        # Remove the 'with' statement and create the client directly
        client = RESTClient(api_key)
        
        print("Connection successful!")
        
        ticker = "QQQ"
        to_date = date.today()
        from_date = to_date - timedelta(days=5) # Fetch a few more days to be safe

        print(f"Fetching data for {ticker}...")
        
        resp = client.get_aggs(
            ticker=ticker,
            multiplier=1,
            timespan="hour",
            from_=from_date.strftime("%Y-%m-%d"),
            to=to_date.strftime("%Y-%m-%d")
        )
        
        df = pd.DataFrame(resp)

        if not df.empty:
            print("\n✅ SUCCESS! Data fetched successfully.")
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            print("Here are the last 5 data points:")
            print(df.tail())
        else:
            print("\n❌ FAILED: Connected successfully, but no data was returned.")
                
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")

if __name__ == "__main__":
    check_polygon_connection()