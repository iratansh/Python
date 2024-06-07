import React, { useState, useEffect } from 'react';
import axios from 'axios';
import StockPredictionImage from './StockPredictionImage';

const SendRequestToPython = ({ stock }) => {
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [stockSymbol, setStockSymbol] = useState('');
  const [showImage, setShowImage] = useState(false);

  const fetchPrediction = async () => {
    setLoading(true);
    setError(null);
    setPrediction(null);
    setShowImage(false);

    let symbol = stock;
    if (typeof stock !== 'string') {
      if (stock instanceof Set) {
        symbol = Array.from(stock).join('');
      } else if (typeof stock === 'object' && stock !== null) {
        symbol = JSON.stringify(stock);  
      } else {
        symbol = String(stock);
      }
    }

    setStockSymbol(symbol);
    console.log('Fetching prediction for stock:', symbol);

    try {
      const response = await axios.get(`http://localhost:5001/predict?stock=${encodeURIComponent(symbol)}`);
      setPrediction(response.data.prediction);
      setLoading(false);
      setTimeout(() => {
        setShowImage(true);
      }, 2000);  // Delay of 2 seconds 
    } catch (error) {
      console.error('Error fetching prediction:', error);
      setError(error.message);
      setLoading(false);
    }
  };

  useEffect(() => {
    if (stock) {
      fetchPrediction();
    }
  }, [stock]);

  return (
    <div>
      <h1 className="text-3xl font-bold mb-4 text-gray-700 header">Predicting {stockSymbol}...</h1>
      {loading && <p className="text-gray-700">Please wait while we fetch the prediction for {stockSymbol}.</p>}
      {error && <p className="text-red-500">Error: {error}</p>}
      {prediction && !error && (
        <div>
          <h2 className="text-2xl font-bold mb-4 text-gray-700">Prediction Result</h2>
          <p className="text-gray-700">The predicted prices for {stockSymbol} for the next 7 trading days are {prediction.join(', ')}.</p>
        </div>
      )}
      {showImage && prediction && !error && <StockPredictionImage stock={stockSymbol} />}
    </div>
  );
};

export default SendRequestToPython;






