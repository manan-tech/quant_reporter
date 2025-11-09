import pandas as pd
import yfinance as yf

def get_data(tickers, start_date, end_date):
    """
    Fetches adjusted close data for a list of tickers, handling
    mismatched trading days and missing data.
    """
    print(f"Fetching data for {', '.join(tickers)}...")
    try:
        all_data = yf.download(tickers, start=start_date, end=end_date)
        
        if all_data.empty:
            raise ValueError("No data downloaded. Check tickers and date range.")

        data = all_data['Close']
        
        if isinstance(data, pd.Series):
            data = data.to_frame(name=tickers[0])
            
        # Forward-fill and back-fill to handle holidays/mismatched calendars
        data_filled = data.ffill().bfill()

        # We must drop any rows that are *still* NaN
        # (e.g., assets that didn't exist at the start)
        data_filled = data_filled.dropna()

        return data_filled
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None