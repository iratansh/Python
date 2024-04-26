"""
Data Analysis for historic stock adj close prices
Author: Ishaan Ratanshi
"""

import pandas as pd
from matplotlib import pyplot as plt
from pylab import rcParams

def usdDateGraph(csv):
    """
    Create a graph that displays Adj Close (USD) as a function of Date
    Input: Csv file with historical stock data
    Returns: None
    """
    df = pd.read_csv(csv, sep=',', parse_dates=['Date'])
    df.loc[:, 'Date'] = pd.to_datetime(df['Date'],format='%Y-%m-%d')
    df.columns = [str(x).lower().replace(' ', '_') for x in df.columns]
    df['month'] = df['date'].dt.month  # Get month of each sample
    df.sort_values(by='date', inplace=True, ascending=True)  # Sort by datetime
    rcParams['figure.figsize'] = 10, 8 # width 10, height 8
    ax = df.plot(x='date', y='adj_close', style='b-', grid=True)
    ax.set_xlabel("date")
    ax.set_ylabel("USD")
    plt.show()

def main():
    """
    Main program function
    Inputs: None
    Returns: None
    """
    usdDateGraph('Stock Data/AAPL.csv')
    usdDateGraph('Stock Data/GOOGL.csv')
    usdDateGraph('Stock Data/MSFT.csv')
    usdDateGraph('Stock Data/TSLA.csv')

main()