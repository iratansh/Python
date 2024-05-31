import React, { useState, useEffect } from 'react';
import axios from 'axios';

const SendRequestToPython = ({ stock }) => {
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchPrediction = async () => {
    setLoading(true);
    setError(null);
    setPrediction(null);
  
    try {
      const response = await axios.get(`http://localhost:5001/predict?stock=${stock}`);
      setPrediction(response.data.prediction);
    } catch (error) {
      console.error('Error fetching prediction:', error);
      setError(error.message);
    } finally {
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
      <h1 className="text-3xl font-bold mb-4 text-gray-700 header">Predicting {stock}...</h1>
      {loading && <p className="text-gray-700">Please wait while we fetch the prediction for {stock}.</p>}
      {error && <p className="text-red-500">Error: {error}</p>}
      {prediction && (
        <div>
          <h2 className="text-2xl font-bold mb-4 text-gray-700">Prediction Result</h2>
          <p className="text-gray-700">The predicted prices for {stock} for the next 7 trading days is {prediction}.</p>
        </div>
      )}
    </div>
  );
};

export default SendRequestToPython;




