"""
Backend App used for SwiftUI app
Predict stock prices using XGBoost and BNN models
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from concurrent.futures import ThreadPoolExecutor
from ComparisonBetweenModels import plot_results, average_parallel_lists
from XGBoost import StockPredictorXGBoost
from BNN import StockPredictorBNN, BNN
import matplotlib.pyplot as plt
import optuna
from NewData import add_historical_stock_data_to_csv
import os
import traceback
import matplotlib

matplotlib.use('Agg')

app = Flask(__name__)
CORS(app)

executor = ThreadPoolExecutor(max_workers=5)

def long_running_task(stock):
    try:
        add_historical_stock_data_to_csv(stock)
        stock_csv_path = f'Stock Data/{stock}.csv'

        if not os.path.exists(stock_csv_path):
            return {'error': f'Stock data file {stock_csv_path} does not exist'}

        stock_predictor_XGBoost = StockPredictorXGBoost(stock_csv_path)
        best_params = stock_predictor_XGBoost.optimize_hyperparameters(n_trials=30)
        model = stock_predictor_XGBoost.train_with_optimal_hyperparameters(best_params)
        next_week_prices_XGBoost = stock_predictor_XGBoost.predict_next_week_close(model)

        stock_predictor_BNN = StockPredictorBNN(stock_csv_path)
        study = optuna.create_study(direction='minimize')
        study.optimize(stock_predictor_BNN.objective, n_trials=50, timeout=3600)

        trial = study.best_trial
        best_hid_dim = trial.params['hid_dim']
        best_prior_scale = trial.params['prior_scale']
        best_lr = trial.params['lr']

        best_model = BNN(in_dim=7, hid_dim=best_hid_dim, prior_scale=best_prior_scale)
        best_guide = stock_predictor_BNN.get_guide(best_model)
        best_svi = stock_predictor_BNN.train(best_model, best_guide, num_iterations=1000, lr=best_lr)

        next_week_prices_BNN = stock_predictor_BNN.predict_next_week_close(best_model, best_guide)

        prediction_avg = average_parallel_lists(next_week_prices_XGBoost, next_week_prices_BNN)

        graph_path = f'./public/{stock}_predictions.png'
        if os.path.exists(graph_path):
            os.remove(graph_path)

        plt.figure(figsize=(12, 6))
        plt.plot(range(1, len(next_week_prices_BNN) + 1), next_week_prices_BNN, label='BNN Predictions', color='red')
        plt.plot(range(1, len(next_week_prices_XGBoost) + 1), next_week_prices_XGBoost, label='XGBoost Predictions', color='green')
        plt.plot(range(1, len(prediction_avg) + 1), prediction_avg, label='Averaged Predictions', color='purple')

        plt.title(f'Stock Price Predictions of Different Models for {stock}')
        plt.xlabel('Day')
        plt.ylabel('Adjusted Closing Price')
        plt.legend()
        plt.savefig(graph_path)
        plt.close()

        prediction_avg = [round(price, 2) for price in prediction_avg]

        return {'prediction': prediction_avg, 'graph_path': f'/graph/{stock}'}
    except Exception as e:
        print(f'Error during prediction: {e}')
        print(traceback.format_exc())
        return {'error': str(e)}

@app.route('/predict', methods=['GET'])
def predict():
    stock = request.args.get('stock')
    if not stock:
        return jsonify({'error': 'Stock name is required'}), 400

    stock = str(stock)
    print(f'Received request for stock: {stock}')

    future = executor.submit(long_running_task, stock)
    try:
        result = future.result(timeout=3600) # make timeout an hour 
        print(f'Sending response: {result}')
        return jsonify(result)
    except Exception as e:
        print(f'Error during prediction: {e}')
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/graph/<stock>', methods=['GET'])
def get_graph(stock):
    graph_path = f'./public/{stock}_predictions.png'
    if os.path.exists(graph_path):
        return send_file(graph_path, mimetype='image/png')
    else:
        return jsonify({'error': 'Graph not found'}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
