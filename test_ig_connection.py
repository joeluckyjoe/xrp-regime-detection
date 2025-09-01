import pandas as pd
from trading_ig import IGService
from datetime import datetime, timedelta

# --- CONFIGURATION ---
IG_USERNAME = "DIGAPIKEY"
IG_PASSWORD = "NmyStanfordpw11V$"
IG_API_KEY = "3b3ca2adfed6a0ed58350939ac9857849c59d259"
ACC_TYPE = "DEMO"

def check_ig_connection():
    """Connects to IG, finds the NASDAQ epic, and fetches data."""
    try:
        print("Connecting to IG Demo Account...")
        ig_service = IGService(IG_USERNAME, IG_PASSWORD, IG_API_KEY, ACC_TYPE)
        ig_service.create_session()
        print("Connection successful!")

        print("\nSearching for the NASDAQ epic...")
        search_result = ig_service.search_markets("NASDAQ")
        
        if not search_result.empty:
            nasdaq_epic = search_result.iloc[0]['epic']
            print(f"Found NASDAQ epic: {nasdaq_epic}")
            
            print(f"\nFetching one day of data for {nasdaq_epic}...")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=1)
            
            data = ig_service.fetch_historical_prices_by_epic_and_date_range(
                epic=nasdaq_epic,
                resolution='5T', # Using the library's preferred name
                start_date=start_date,
                end_date=end_date
            )
            
            # --- THIS IS THE FIX ---
            # Check if the returned data is a DataFrame before checking if it's empty
            if isinstance(data, pd.DataFrame) and not data.empty:
                print("\n✅ SUCCESS! Data fetched successfully.")
                print("Here are the last 5 data points:")
                print(data.tail())
            else:
                print("\n❌ FAILED: Connected successfully, but no data was returned.")
                print("API Response:", data) # Print the error dictionary from the API
        else:
            print("\n❌ FAILED: Could not find the epic for NASDAQ.")
            
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")

if __name__ == "__main__":
    check_ig_connection()