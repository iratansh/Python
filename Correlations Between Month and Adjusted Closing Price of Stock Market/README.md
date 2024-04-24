This Python code analyzes stock data from CSV files using the Pandas library for data manipulation and Matplotlib for visualization. This code helps visualize how the average adjusted closing prices of different stocks vary across months, providing insights into potential seasonal patterns or trends.

Reading Data: The readDataFromFile function reads a CSV file containing stock data, specifically columns for 'Date' and 'Adj Close' (adjusted closing price), and returns these columns as lists.
Calculating Monthly Averages: The find_AVGprice_for_each_month function calculates the average adjusted closing price for each month. It creates a dictionary where each key represents a month, and the value is a list containing the sum of adjusted prices, the number of trading days, and the average price.
Displaying Correlations: The show_correlations function generates a scatter plot showing the average adjusted closing price for each month. It takes a dictionary with monthly data and plots the months on the x-axis and the corresponding average prices on the y-axis.
Main Function: The main function reads stock data from multiple CSV files, calculates monthly averages for each stock, and displays correlations using scatter plots for each stock.

Technologies Used:
Pandas: For reading and manipulating the stock data stored in CSV files.
Matplotlib: For creating visualizations, particularly scatter plots to display correlations between the average adjusted closing price and months.
