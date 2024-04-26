"""
DATA ANALYSIS
Find Correlations between Month and AVG Adjusted Closing Price, Find Correlations between Days in the Month and AVG Adjusted Closing Price
Author: Ishaan Ratanshi
"""

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def readDataFromFile(csv):
    """
    Read data from historic stock files
    Input: Csv filename
    Returns: Date column, adj column
    """
    file = pd.read_csv(csv, usecols=['Date', 'Adj Close'])
    date = file['Date']
    adj = file['Adj Close']
    return date, adj


def find_AVGprice_for_each_month_and_day(dates, adjs):
    """
    Find avg price for each month and day for a stock
    Input: Date column, adj column
    Returns: month price dictionary, day price dictionary
    """
    # First element in list = added adj values for the month and second value = number of tradining days in the month and third value = avg 
    month_price_dict = {'January':[0, 0, 0], 'February':[0, 0, 0], 'March':[0 ,0, 0], 'April':[0, 0, 0], 'May':[0, 0, 0], 'June':[0, 0, 0], 'July':[0, 0, 0], 'August':[0, 0, 0], 'September':[0, 0, 0], 'October':[0, 0, 0], 'November':[0, 0, 0], 'December':[0, 0, 0]}
    day_price_dict = {'Monday':[0, 0, 0], 'Tuesday':[0, 0, 0], 'Wednesday':[0, 0, 0], 'Thursday':[0, 0, 0], 'Friday':[0, 0, 0]}
    
    # FOR MONTH PRICE DICT
    for date, adj in zip(dates, adjs):
        month = datetime(int(date[:4]), int(date[5:7]), int(date[8::])).strftime('%B')
        month_price_dict[month][0] += adj
        month_price_dict[month][1] += 1
    for el in month_price_dict.values():
        el[2] = el[0]/el[1]

    # FOR DAY PRICE DICT
    for date, adj in zip(dates, adjs):
        day = datetime(int(date[:4]), int(date[5:7]), int(date[8::])).strftime('%A')
        day_price_dict[day][0] += adj
        day_price_dict[day][1] += 1
    for el in day_price_dict.values():
        el[2] = el[0]/el[1]

    return month_price_dict, day_price_dict
    

def show_correlations_between_months(month_price_dict):
    """
    Show correlations between Months and adj price
    Input: Month price dictionary
    Returns: None
    """
    title = 'Adjusted Closing Price as a Function of Month'
    month = list(month_price_dict.keys())
    avg = list()
    for el in month_price_dict.values():
        avg.append(el[2])
    plt.scatter(month, avg)
    plt.title(title, loc='center')
    plt.show()

def show_correlations_between_days_of_month(day_price_dict):
    """
    Show correlations between weekdays and adj price
    Input: Day price dictionary
    Returns: None
    """
    title = 'Adjusted Closing Price as a Function of Weekday'
    day = list(day_price_dict.keys())
    avg = list()
    for el in day_price_dict.values():
        avg.append(el[2])
    plt.scatter(day, avg)
    plt.title(title, loc='center')
    plt.show()


def main():
    """
    Main program function
    Inputs: None
    Returns: None
    """
    date_AAPL, adj_AAPL = readDataFromFile('Stock Data/AAPL.csv')
    date_GOOGL, adj_GOOGL = readDataFromFile('Stock Data/GOOGL.csv')
    date_MSFT, adj_MSFT = readDataFromFile('Stock Data/MSFT.csv')
    date_SPOT, adj_SPOT = readDataFromFile('Stock Data/SPOT.csv')
    date_TSLA, adj_TSLA = readDataFromFile('Stock Data/TSLA.csv')
    month_dict_forAAPL, day_dict_forAAPL = find_AVGprice_for_each_month_and_day(date_AAPL, adj_AAPL)
    month_dict_forGOOGL, day_dict_forGOOGL = find_AVGprice_for_each_month_and_day(date_GOOGL, adj_GOOGL)
    month_dict_forMSFT, day_dict_forMSFT = find_AVGprice_for_each_month_and_day(date_MSFT, adj_MSFT)
    month_dict_forSPOT, day_dict_forSPOT = find_AVGprice_for_each_month_and_day(date_SPOT, adj_SPOT)
    month_dict_forTSLA, day_dict_forTSLA = find_AVGprice_for_each_month_and_day(date_TSLA, adj_TSLA)

    show_correlations_between_months(month_dict_forAAPL)  # Crash in May
    show_correlations_between_months(month_dict_forGOOGL)  # Crash in May
    show_correlations_between_months(month_dict_forMSFT)  # Crash in May
    show_correlations_between_months(month_dict_forSPOT)  # Crash in May
    show_correlations_between_months(month_dict_forTSLA)  # Crash in May

    show_correlations_between_days_of_month(day_dict_forAAPL)  # Peak at Wednesday
    show_correlations_between_days_of_month(day_dict_forGOOGL)  # Peak at Wednesday / Thursday
    show_correlations_between_days_of_month(day_dict_forMSFT)  # Peak at Wednesday
    show_correlations_between_days_of_month(day_dict_forSPOT)  # Inverse Relationship
    show_correlations_between_days_of_month(day_dict_forTSLA)  # Peak at Wednesday


main()
