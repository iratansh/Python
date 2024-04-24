"""
Find Correlations between Month and Adjusted Closing Price
Author: Ishaan Ratanshi
"""

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def readDataFromFile(csv):
    file = pd.read_csv(csv, usecols=['Date', 'Adj Close'])
    date = file['Date']
    adj = file['Adj Close']
    return date, adj


def find_AVGprice_for_each_month(dates, adjs):
    # First element in list = added adj values for the month and second value = number of tradining days in the month and third value = avg 
    month_price_dict = {'January':[0, 0, 0], 'February':[0, 0, 0], 'March':[0 ,0, 0], 'April':[0, 0, 0], 'May':[0, 0, 0], 'June':[0, 0, 0], 'July':[0, 0, 0], 'August':[0, 0, 0], 'September':[0, 0, 0], 'October':[0, 0, 0], 'November':[0, 0, 0], 'December':[0, 0, 0]}
    for date, adj in zip(dates, adjs):
        month = datetime(int(date[:4]), int(date[5:7]), int(date[8::])).strftime('%B')
        month_price_dict[month][0] += adj
        month_price_dict[month][1] += 1
    for el in month_price_dict.values():
        el[2] = el[0]/el[1]
    return month_price_dict
    

def show_correlations(month_price_dict):
    title = 'Adjusted Closing Price as a Function of Month'
    month = list(month_price_dict.keys())
    avg = list()
    for el in month_price_dict.values():
        avg.append(el[2])
    plt.scatter(month, avg)
    plt.title(title, loc='center')
    plt.show()


def main():
    date_AAPL, adj_AAPL = readDataFromFile('Stock Data/AAPL.csv')
    date_GOOGL, adj_GOOGL = readDataFromFile('Stock Data/GOOGL.csv')
    date_MSFT, adj_MSFT = readDataFromFile('Stock Data/MSFT.csv')
    date_SPOT, adj_SPOT = readDataFromFile('Stock Data/SPOT.csv')
    dict_forAAPL = find_AVGprice_for_each_month(date_AAPL, adj_AAPL)
    dict_forGOOGL = find_AVGprice_for_each_month(date_GOOGL, adj_GOOGL)
    dict_forMSFT = find_AVGprice_for_each_month(date_MSFT, adj_MSFT)
    dict_forSPOT = find_AVGprice_for_each_month(date_SPOT, adj_SPOT)

    show_correlations(dict_forAAPL)
    show_correlations(dict_forGOOGL)
    show_correlations(dict_forMSFT)
    show_correlations(dict_forSPOT)

main()