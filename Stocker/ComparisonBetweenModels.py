import matplotlib.pyplot as plt

def plot_results(y_test, y_pred_bnn, y_pred_xgb, y_pred_avg):
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(y_test) + 1), y_test, label='Actual Prices', color='blue')
    plt.plot(range(1, len(y_pred_bnn) + 1), y_pred_bnn, label='BNN Predictions', color='red')
    plt.plot(range(1, len(y_pred_xgb) + 1), y_pred_xgb, label='XGBoost Predictions', color='green')
    plt.plot(range(1, len(y_pred_avg) + 1), y_pred_avg, label='Averaged Predictions', color='purple')
    
    plt.title('Stock Price Predictions vs. Actual Prices')
    plt.xlabel('Day')
    plt.ylabel('Adjusted Closing Price')
    plt.legend()
    plt.show()

actual_prices = [168.79, 169.66, 169.07, 173.26, 170.10, 169.07, 172.80]
pred_bnn = [178.30183, 178.57834, 180.0635, 182.80518, 180.9727, 182.26105, 184.41537]
pred_xgb = [163.73656, 163.65869, 165.00328, 164.54767, 163.89056, 162.7681, 162.00394]
pred_avg = [171.019195, 171.118515, 172.53339, 173.676425, 172.43163, 172.514575, 173.209655]

plot_results(actual_prices, pred_bnn, pred_xgb, pred_avg)
