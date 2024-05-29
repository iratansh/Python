Citations: https://forecastegy.com/posts/xgboost-hyperparameter-tuning-with-optuna/, https://xgboost.readthedocs.io/en/stable/

This code uses the XGBoost machine learning algorithm to predict stock market prices. It includes steps for reading and processing historical stock data, scaling the data, and training the model. Additionally, it uses Optuna to optimize hyperparameters for the XGBoost model, enhancing its prediction accuracy. By integrating Optuna for hyperparameter optimization, the code aims to improve the prediction accuracy of the XGBoost model for stock market prices.

Technologies Used:

* XGBoost: A powerful and efficient gradient boosting framework used for regression tasks.
* Optuna: An automatic hyperparameter optimization framework to improve model performance.
* Pandas: A data manipulation and analysis library.
* NumPy: A library for numerical operations.
* Scikit-learn: Used for preprocessing data and calculating performance metrics.
* Matplotlib: A plotting library to visualize results.
* Datetime: Python library to handle date and time operations.
  
Data Reading and Preprocessing:
* read_data_from_file: Reads stock data from a CSV file.
* process_dataframe: Creates lagged features for time series prediction.
* scale_dataframe: Scales the data using MinMaxScaler for better model performance.

Data Splitting:
* split_data: Splits the scaled data into features (X) and target (y).
* split_data_into_train_test: Further splits the data into training and testing sets.

Model Training and Evaluation:
* train_and_evaluate: Trains the XGBoost model and evaluates its performance using RMSE and MAPE.


Hyperparameter Optimization:
* objective: Defines the objective function for Optuna to optimize. It includes various hyperparameters like n_estimators, learning_rate, max_depth, etc.
* Optuna Study: Conducts the optimization study to find the best hyperparameters.

Prediction Functions:
* predict_next_day_close: Predicts the next day's adjusted closing price.
* predict_next_week_close: Predicts the adjusted closing prices for the next week.

Main Execution:
* main: The main function that integrates all steps:
* Reads and processes the data.
* Scales the data.
* Splits the data into training and testing sets.
* Performs hyperparameter optimization using Optuna.
* Trains the final model using the best hyperparameters.
* Predicts the next day's and the next week's adjusted closing prices.
* (Optionally) Plots and saves the results.

