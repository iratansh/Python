"""
This module contains functions to add any new stock data to a CSV file for a given stock.
"""

import yfinance as yf
import pandas as pd
import os

def format_date_column(df):
    """
    Ensure the 'Date' column contains only the date part and has the correct column label.
    Input: df (pd.DataFrame) - DataFrame with stock data
    Output: df (pd.DataFrame) - DataFrame with formatted 'Date' column
    """
    df.reset_index(inplace=True) # Reset the index
    df['Date'] = pd.to_datetime(df['Date']).dt.date  # Convert the 'Date' column to date only
    df.set_index('Date', inplace=True)  # Set the 'Date' column as the index
    df.index.name = 'Date'  # Rename the index
    return df

def add_historical_stock_data_to_csv(stock):
    """
    Add historical stock data to a CSV file for a given stock.
    If the CSV file already exists, append new data to it.
    Input: stock (str) - Stock ticker symbol
    Output: None
    """
    csv_file = f"Stock Data/{stock}.csv"
    
    # Load existing data if the file exists
    if os.path.exists(csv_file):
        existing_data = pd.read_csv(csv_file, parse_dates=['Date'])
        existing_data.set_index('Date', inplace=True)
        existing_data = format_date_column(existing_data)
        last_date = existing_data.index[-1]
        print(f"Existing data up to: {last_date}")
    else:
        existing_data = pd.DataFrame()
        last_date = None
   
    stock_data = yf.Ticker(stock)  # Get stock data from Yahoo Finance
    
    if last_date:
        start_date = last_date + pd.Timedelta(days=1)
        new_data = stock_data.history(start=start_date)
    else:
        new_data = stock_data.history(period='5y')
    
    if not new_data.empty:
        # Process the new data
        new_data.reset_index(inplace=True)
        new_data['Date'] = new_data['Date'].dt.date
        new_data = new_data[['Date', 'Open', 'High', 'Low', 'Close']]
        new_data = new_data.round({'Open': 5, 'High': 5, 'Low': 5, 'Close': 5})
        new_data = new_data.rename(columns={'Close': 'Adj Close'})
        new_data.set_index('Date', inplace=True)

        # Combine the existing and new data
        combined_data = pd.concat([existing_data, new_data])
        combined_data = combined_data[~combined_data.index.duplicated(keep='last')]  # Remove duplicate rows
        combined_data = combined_data[['Open', 'High', 'Low', 'Adj Close']]

        # Write the combined data to a temporary file first
        temp_csv_file = f"{csv_file}.tmp"
        combined_data.to_csv(temp_csv_file, date_format='%Y-%m-%d')

        os.replace(temp_csv_file, csv_file)  # Replace the old file with the new file

        print(f"Data appended and saved to {csv_file}")
    else:
        print("No new data available to append.")


















