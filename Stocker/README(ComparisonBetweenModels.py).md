Results (Stock == AAPL from April 24, 2024 - May 2, 2024) with Optuna Tuning:

* Actual: Predicted next week adjusted closing prices: [168.79, 169.66, 169.07, 173.26, 170.10, 169.07, 172.80]
* BNN: Predicted next week adjusted closing prices: [163.73656, 163.65869, 165.00328, 164.54767, 163.89056, 162.7681, 162.00394]
* XGBoost: Predicted next week adj closing prices: [178.30183, 178.57834, 180.0635, 182.80518, 180.9727, 182.26105, 184.41537] 
* Averaged BNN and XGBoost: Predicted next week adj closing prices: [171.019195, 171.118515, 172.53339, 173.676425, 172.43163, 172.514575, 173.209655]
* Percentage Error for BNN: [2.993921440843649, 3.5372568666745194, 2.4053468977346686, 5.028471661087372, 3.6504644326866558, 3.7273910214703907, 6.247719907407413]
* Percentage Error for XGBoost: [5.6353042241838995, 5.256595544029236, 6.502336310403982, 5.5091654161376065, 6.391945914168142, 7.8021233808481805, 6.7218576388888795] 
* Percentage Error for Averaged BNN and XGBoost: [1.3206913916701253, 0.859669338677358, 2.048494706334657, 0.240346877525109, 1.3707407407407515, 2.037366179688895, 0.23706886574073296]

Analysis:

Bayesian Neural Network (BNN)
* Strengths: Captures broader trends and is responsive to changes in stock prices.
* Weaknesses: Typically underestimates the closing prices, leading to lower predicted values compared to the actual prices.  

XGBoost
* Strengths: Produces conservative and more stable predictions.
* Weaknesses: Typically overestimates the closing prices, resulting in a higher predicted trend compared to the actual prices.

Averaged Predictions
* Strengths: Provides a balanced approach by combining the strengths of both models, resulting in closer approximations to the actual prices.
* Weaknesses: Still shows some deviation from the actual prices but improves overall prediction accuracy.

Conclusion
The BNN model typically tends to underpredict while the XGBoost model tends to overpredict the adjusted closing prices. Averaging the results of both models offers a more balanced prediction, yielding a closer approximation to the actual stock prices.

<img width="1197" alt="image" src="https://github.com/iratansh/Python/assets/151393106/af1dadcb-551c-4abb-bfd9-64f337a1a061">


