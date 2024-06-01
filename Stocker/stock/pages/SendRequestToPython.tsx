import React, { useState, useEffect } from 'react';
import axios from 'axios';
import StockPredictionImage from './StockPredictionImage';

const SendRequestToPython = ({ stock }) => {
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showGraph, setShowGraph] = useState(false);

  const fetchPrediction = async () => {
    setLoading(true);
    setError(null);
    setPrediction(null);
    setShowGraph(false);

    let stockSymbol = stock;
    if (typeof stock !== 'string') {
      if (stock instanceof Set) {
        stockSymbol = Array.from(stock).join('');
      }
    }

    console.log('Fetching prediction for stock:', stockSymbol);

    try {
      const response = await axios.get(`http://localhost:5001/predict?stock=${encodeURIComponent(stockSymbol)}`);
      setPrediction(response.data.prediction);
    } catch (error) {
      console.error('Error fetching prediction:', error);
      setError(error.message);
    } finally {
      setLoading(false);
      setTimeout(() => setShowGraph(true), 1000);  // 1-second delay
    }
  };

  useEffect(() => {
    if (stock) {
      fetchPrediction();
    }
  }, [stock]);

  return (
    <div>
      <h1 className="text-3xl font-bold mb-4 text-gray-700 header">Predicting {stock}...</h1>
      {loading && <p className="text-gray-700">Please wait while we fetch the prediction for {stock}.</p>}
      {error && <p className="text-red-500">Error: {error}</p>}
      {prediction && (
        <div>
          <h2 className="text-2xl font-bold mb-4 text-gray-700">Prediction Result</h2>
          <p className="text-gray-700">The predicted prices for {stock} for the next 7 trading days are {prediction.join(', ')}.</p>
        </div>
      )}
      {showGraph && <StockPredictionImage stock={stock} />}
    </div>
  );
};

export default SendRequestToPython;





