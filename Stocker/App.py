"""
Backend App 
Predicts next 7 days adj close prices using BNN and XGBoost and relays the results to the frontend
Saves the plot for representation to the public files in frontend
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from ComparisonBetweenModels import plot_results, average_parallel_lists
from XGBoost import StockPredictorXGBoost
from BNN import StockPredictorBNN, BNN
import matplotlib, optuna

matplotlib.use('Agg')

app = Flask(__name__)
CORS(app) 

@app.route('/predict', methods=['GET'])
def predict():
    stock = request.args.get('stock')
    if not stock:
        return jsonify({'error': 'Stock name is required'}), 400

    stock = str(stock)
    print(f'Received request for stock: {stock}')
    
    try:
        # Update the stock data CSV file with the latest data
        add_historical_stock_data_to_csv(stock)
        
        # Define the path to the updated CSV file
        stock_csv_path = f'Stock Data/{stock}.csv'

        # Check if the updated CSV file exists
        if not os.path.exists(stock_csv_path):
            return jsonify({'error': f'Stock data file {stock_csv_path} does not exist'}), 400
        
        # Predict using XGBoost model
        stock_predictor_XGBoost = StockPredictorXGBoost(stock_csv_path)
        best_params = stock_predictor_XGBoost.optimize_hyperparameters(n_trials=30)
        model = stock_predictor_XGBoost.train_with_optimal_hyperparameters(best_params)
        next_week_prices_XGBoost = stock_predictor_XGBoost.predict_next_week_close(model)
        
        # Predict using BNN model
        stock_predictor_BNN = StockPredictorBNN(stock_csv_path)
        study = optuna.create_study(direction='minimize')  # Create an Optuna study and optimize the objective function
        study.optimize(stock_predictor_BNN.objective, n_trials=50)
        trial = study.best_trial

        # Train the BNN model with the best hyperparameters
        best_hid_dim = trial.params['hid_dim']
        best_prior_scale = trial.params['prior_scale']
        best_lr = trial.params['lr']
        best_model = BNN(in_dim=7, hid_dim=best_hid_dim, prior_scale=best_prior_scale)
        best_guide = stock_predictor_BNN.get_guide(best_model)
        best_svi = stock_predictor_BNN.train(best_model, best_guide, num_iterations=1000, lr=best_lr)
        y_pred_mean, y_pred_std = stock_predictor_BNN.predict(best_model, best_guide, stock_predictor_BNN.X_test_tensor)  # Predict on the test set using the best model
        next_week_prices_BNN = stock_predictor_BNN.predict_next_week_close(best_model, best_guide)  # Predict the adjusted closing price

        # Average the predictions from both models
        prediction_avg = average_parallel_lists(next_week_prices_XGBoost, next_week_prices_BNN)
        
        # Plot the results (optional, might need to adjust the function call as needed)
        actual_prices = [168.79, 169.66, 169.07, 173.26, 170.10, 169.07, 172.80]  # Replace with actual prices if available
        plot_results(stock, actual_prices, next_week_prices_BNN, next_week_prices_XGBoost, prediction_avg)
        prediction_avg = [round(price, 2) for price in prediction_avg]

        return jsonify({'prediction': prediction_avg})
    except Exception as e:
        print(f'Error during prediction: {e}')
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
