References: Chandra R , Jain K , Deo R V , et al. Langevin-gradient parallel tempering for Bayesian neural learning[J]. Neurocomputing, 2019, 359(SEP.24):315-326., https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/DL2/Bayesian_Neural_Networks/dl2_bnn_tut1_students_with_answers.html

This program is for stock time series forecasting using a Bayesian Neural Network (BNN) implemented with Pyro, a probabilistic programming library built on top of PyTorch. 
The main function orchestrates the entire process, including hyperparameter optimization, model training, evaluation, and prediction. It uses Optuna to search for the best hyperparameters, trains the model with those hyperparameters, evaluates on the test set, plots the results, and finally predicts future stock prices.

1. **Bayesian Neural Network (BNN)**:
   - The `BNN` class defines the architecture of the Bayesian neural network model. It has fully connected layers with a tanh activation function.
   - Pyro is used to define the probabilistic layers (`PyroModule`) and sample weights and biases from prior distributions.

2. **StockPredictor**:
   - This class handles data preprocessing, model training, and prediction tasks.
   - It reads stock data from a CSV file and processes it to create features for prediction.
   - `MinMaxScaler` from scikit-learn is used for feature scaling.
   - Data is split into training and testing sets.
   - PyTorch tensors are created for input data.
   - Functions are provided to train the BNN model, make predictions, and plot results.
   - Methods are also included to predict the next day's and next week's closing prices.

3. **Technologies**:
   - **PyTorch**: For defining and training neural network models.
   - **Pyro**: For probabilistic modeling and Bayesian inference.
   - **scikit-learn**: For data preprocessing, specifically `MinMaxScaler`.
   - **pandas**: For data manipulation, particularly for reading and processing CSV files.
   - **matplotlib**: For plotting the stock price predictions.
   - **optuna**: For hyperparameter optimization using Bayesian optimization.

