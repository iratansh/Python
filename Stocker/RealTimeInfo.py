"""
Get latest stock data to add to current datasets
"""

import yfinance as yf
import pandas as pd
import os

def add_historical_stock_data_to_csv(stock):
    """
    Add historical stock data to a CSV file for a given stock.
    If the CSV file already exists, append new data to it.
    Input: stock (str) - Stock ticker symbol
    Output: None
    """
    csv_file = f"{stock}.csv"
    
    if os.path.exists(csv_file):
        # Load existing data
        existing_data = pd.read_csv(csv_file, index_col='Date', parse_dates=True)
        last_date = existing_data.index[-1]
        print(f"Existing data up to: {last_date}")
    else:
        # No existing data
        existing_data = pd.DataFrame()
        last_date = None
        
    stock_data = yf.Ticker(stock)  # Fetch new data from Yahoo Finance
    if last_date:
        start_date = last_date + pd.Timedelta(days=1)          # Fetch new data from the day after the last date to today
        new_data = stock_data.history(start=start_date)
    else:
        new_data = stock_data.history(period='1mo')  # Fetch data for the last month if there's no existing data
    if not new_data.empty:
        combined_data = pd.concat([existing_data, new_data])   # Combine existing data with new data
        combined_data = combined_data[~combined_data.index.duplicated(keep='last')]  # Remove duplicates, if any
        combined_data.to_csv(csv_file)  # Save combined data to CSV
        print(f"Data appended and saved to {csv_file}")
    else:
        print("No new data available to append.")
