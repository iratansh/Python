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
    if not stock:
        return jsonify({'error': 'Stock name is required'}), 400

    stock = str(stock)
    print(f'Received request for stock: {stock}')

    try:
        # Predict using XGBoost model
        stock_predictor_XGBoost = StockPredictorXGBoost(f'Stock Data/{stock}.csv')
        best_params = stock_predictor_XGBoost.optimize_hyperparameters(n_trials=30)
        xgboost_model = stock_predictor_XGBoost.train_with_optimal_hyperparameters({'n_estimators': 394, 'learning_rate': 0.06893921597530149, 'max_depth': 9, 'subsample': 0.9129917307049513, 'colsample_bytree': 0.7942234675042359, 'min_child_weight': 8})
        next_week_prices_XGBoost = stock_predictor_XGBoost.predict_next_week_close(xgboost_model)

        # Predict using BNN model
        stock_predictor_BNN = StockPredictorBNN(f'Stock Data/{stock}.csv')
        best_hid_dim = 12
        best_prior_scale = 2.0065996491097478
        best_lr = 0.0005877286975873452
        best_model = BNN(in_dim=7, hid_dim=best_hid_dim, prior_scale=best_prior_scale)
        best_guide = stock_predictor_BNN.get_guide(best_model)
        stock_predictor_BNN.train(best_model, best_guide, num_iterations=1000, lr=best_lr)
        
        # Predict the next week adjusted closing prices
        next_week_prices_BNN = stock_predictor_BNN.predict_next_week_close(best_model, best_guide)

        # Average the predictions from both models
        prediction_avg = average_parallel_lists(next_week_prices_XGBoost, next_week_prices_BNN)
        
        # Plot the results (optional, might need to adjust the function call as needed)
        actual_prices = [168.79, 169.66, 169.07, 173.26, 170.10, 169.07, 172.80]
        plot_results(stock, actual_prices, next_week_prices_BNN, next_week_prices_XGBoost, prediction_avg)
        prediction_avg = [round(price, 2) for price in prediction_avg]

        return jsonify({'prediction': prediction_avg})
    except Exception as e:
        print(f'Error during prediction: {e}')
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
