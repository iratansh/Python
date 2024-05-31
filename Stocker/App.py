import flask
from flask import Flask, request, jsonify
from flask_cors import CORS
from ComparisonBetweenModels import plot_results, average_parallel_lists
from XGBoost import StockPredictorXGBoost
from BNN import StockPredictorBNN, BNN
import matplotlib

matplotlib.use('Agg')

app = Flask(__name__)
CORS(app)  

@app.route('/predict', methods=['GET'])
def predict():
    stock = request.args.get('stock')

    # Predict using XGBoost model
    stock_predictor_XGBoost = StockPredictorXGBoost('Stock Data/AAPL.csv')
    best_params = stock_predictor_XGBoost.optimize_hyperparameters(n_trials=30)
    xgboost_model = stock_predictor_XGBoost.train_with_optimal_hyperparameters(best_params)
    # Predict the next week adjusted closing prices
    next_week_prices_XGBoost = stock_predictor_XGBoost.predict_next_week_close(xgboost_model)

    # Predict using BNN model
    stock_predictor_BNN = StockPredictorBNN('Stock Data/AAPL.csv')
    best_hid_dim = 13
    best_prior_scale = 2.5788713266041654
    best_lr = 0.0045095808163138
    best_model = BNN(in_dim=7, hid_dim=best_hid_dim, prior_scale=best_prior_scale)
    best_guide = stock_predictor_BNN.get_guide(best_model)
    stock_predictor_BNN.train(best_model, best_guide, num_iterations=1000, lr=best_lr)
    
    next_week_prices_BNN = stock_predictor_BNN.predict_next_week_close(best_model, best_guide)
    # Average the predictions from both models
    prediction_avg = average_parallel_lists(next_week_prices_XGBoost, next_week_prices_BNN)
    
    # Plot the results (optional, might need to adjust the function call as needed)
    actual_prices = [168.79, 169.66, 169.07, 173.26, 170.10, 169.07, 172.80]
    plot_results(actual_prices, next_week_prices_BNN, next_week_prices_XGBoost, prediction_avg)
    prediction_avg = [round(price, 2) for price in prediction_avg]

    return jsonify({'prediction': prediction_avg})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
