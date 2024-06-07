import React, { useEffect, useState } from 'react';

const StockPredictionImage = ({ stock }) => {
  const [imageSrc, setImageSrc] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    if (stock) {
      setLoading(true);
      const imagePath = `./${stock}_predictions.png`;  
      fetch(imagePath)
        .then(response => {
          if (response.ok) {
            setImageSrc(imagePath);
            setLoading(false);
          } else {
            throw new Error('Image not found');
          }
        })
        .catch(error => {
          console.error('Error fetching image:', error);
          setError('Error fetching image');
          setLoading(false);
        });
    }
  }, [stock]);

  return (
    <div>
      {loading && <p>Loading...</p>}
      {error && <p>{error}</p>}
      {<img src={imageSrc} alt={`Stock Predictions for ${stock}`} />}
    </div>
  );
};

export default StockPredictionImage;
